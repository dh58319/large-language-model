## reference : https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py

import argparse
import json
import logging
import math
import os
import wandb
from itertools import chain
from datetime import timedelta

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

## import model & model configuration
from transformers.models.bert.modeling_bert import BertForMaskedLM as scratch_model
from transformers.models.bert.configuration_bert import BertConfig

def set_config(args):
    return BertConfig()
    # return BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, attention_probs_dropout_prob=args.drop_prob)

## Error will be occured if minimal version of Transformers is not installed
check_min_version("4.38.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

parser = argparse.ArgumentParser(description="Pretraining on Masked Language Modeling task - Pytorch Large Language Model")

group = parser.add_argument_group('Dataset')
group.add_argument('--data_dir', type=str, help="Directory of stored Dataset")
group.add_argument('--dataset', type=str, help="The name of the dataset")
group.add_argument('--dataset_config', type=str, help="The configuration name of the dataset")
group.add_argument('-s', '--streaming', action='store_true', default=False,
                   help="Enable streaming mode -> not require local disk usage")
group.add_argument('--train_file', type=str,
                   help="A file contains the training data -> extension : .csv, .json, .txt")
group.add_argument('--valid_file', type=str,
                   help="A file contains the validation data -> extension : .csv, .json, .txt")
group.add_argument('--valid_split_percentage', type=int, default=5,
                   help="Percentage of Train set used as Valid set if there is no Valid split")
group.add_argument('--pad_to_max_length', action='store_true', default=False,
                   help="Pad all samples to 'max_length'. Otherwise, dynamic padding is used")
group.add_argument('--max_seq_length', type=int, default=None,
                   help=(
                       "Maximum total input sequence length after tokenization."
                       "Sequences longer than this value will be truncated."
                   ))
group.add_argument('--line_by_line', action='store_true', default=False,
                   help="Distinguish lines of text in Dataset -> Distinct Sequences")

group = parser.add_argument_group('Model')
group.add_argument('--model', type=str, required=False,
                   help="Name or Path of the Pretrained model from the Hub - https://huggingface.co/models")
group.add_argument('--config', type=str,
                   help="Name or Path of the Pretrained config (required if model_name != config_name)")
group.add_argument('--tokenizer', type=str,
                   help="Name or Path of the Pretrained tokenizer (required if model_name != tokenizer_name)")

group = parser.add_argument_group('Parameter')
group.add_argument('--train_batch_size', type=int, default=16,
                   help="Batch size for Training dataloader (per device)")
group.add_argument('--eval_batch_size', type=int, default=16,
                   help="Batch size for evaluation dataloader (per device)")
group.add_argument('--mlm_probability', type=float, default=0.15, help="Probability of masked tokens for MLM")
group.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
group.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
group.add_argument('--train_epoch', type=int, default=3, help="Total number of Training epochs")
group.add_argument('--max_train_steps', type=int, default=None,
                   help="Total number of Training steps -> override 'epochs'")
group.add_argument('--grad_accum_steps', type=int, default=1,
                   help="Number of update steps to accumulate gradients before performing a backward/update pass")
group.add_argument('--sche', type=str, default="linear",
                   help="LR Scheduler : 'linear', 'cosine', cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'")
group.add_argument('--num_warmup_steps', type=int, default=0, help="Number of Warmup step in LR scheduler")
group.add_argument('--drop_prob', type=float, default=0.0, help="Dropout Probability")

group = parser.add_argument_group('Setting')
group.add_argument('--log_wandb', action='store_true', default=False,
                   help="log training and validation metrics to wandb")
group.add_argument('--project', type=str, default='',
                   help="name of train project, name of sub-folder for output")
group.add_argument('--run_name', type=str, default=False, help="run name for wandb")
group.add_argument('--out_dir', type=str, default=None, help="Directory to store the final model")
group.add_argument('--checkpointing_steps', type=str, default=None,
                   help=(
                       " default: not save checkpoint "
                       " 'n'(ex.'10'): save checkpoint for every n(ex.10) step "
                       " 'epoch': save checkpoint for every epoch "
                   ))
group.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help="Directory to continue training from Checkpoint")

parser.add_argument('--seed', type=int, default=None, help="Seed for reproducible training")
parser.add_argument('--preprocess_num_worker', type=int, default=None,
                    help="Number of process for Preprocessing")
parser.add_argument('--overwrite_cache', action="store_true", help="Overwrite the cached Train & Eval sets")
parser.add_argument('--trust_remote_code', action="store_true", default=False,
                    help=(
                        "Allow custom models (defined on the Hub - https://huggingface.co/models)"
                        "true : Trust repositories & Execute code present on the Hub"
                    ))
parser.add_argument('--with_tracking', action="store_true", default=False, help="Enable experiment trackers for Logging")
parser.add_argument('--low_cpu_mem_usage', action="store_true",
                    help=(
                        "Create the model as an empty shell -> only materialize parameters when pretrained models are loaded"
                        "If passed, LLM loading time and RAM comsumption will be benefited"
                    ))

def main(model_config, args):
    ## Initialize the Accelerator
    accelerate_log_kwargs = {}
    if args.with_tracking:
        accelerate_log_kwargs["output_dir"] = args.out_dir
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps, kwargs_handlers=[kwargs], **accelerate_log_kwargs)

    ## Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    ## Set training seed
    if args.seed:
        set_seed(args.seed)

    ## Create Output directory if it does not exist
    if accelerator.is_main_process:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ## Get Dataset
    ## public datasets are available on the hub - https://huggingface.co/datasets/
    if args.dataset:
        ## Store & Load dataset from Hub or Load dataset from initial directory
        raw_datasets = load_dataset(args.dataset, args.dataset_config, streaming=args.streaming, trust_remote_code=True)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset,
                args.dataset_config,
                split=f"train[:{args.valid_split_percentage}%]",
                streaming=args.streaming,
                trust_remote_code=True
            )
            raw_datasets["train"] = load_dataset(
                args.dataset,
                args.dataset_config,
                split=f"train[{args.valid_split_percentage}%:]",
                streaming=args.streaming,
                trust_remote_code=True
            )
    else:
        ## Load dataset from the specific directory
        raw_datasets = load_dataset(args.data_dir)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.data_dir,
                split=f"train[:{args.valid_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.data_dir,
                split=f"train[{args.valid_split_percentage}%:]",
            )

    ## Load Pretrained Model & Tokenizer
    if args.config:
        ## if model_name != config_name
        config = AutoConfig.from_pretrained(args.config, trust_remote_code=args.trust_remote_code)
    elif args.model:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    else:
        ## Train from Scratch -> Load Configuration
        config = model_config
        logger.warning("You are instantiating a new config instance from Scratch")

    if args.tokenizer:
        ## model_name != tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from Scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name"
        )

    if args.model:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model,
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        ## Train from Scratch -> Load Model
        model = scratch_model(config)
        logger.info("Training new model from Scratch")

    ## Resize Embedding size only when necessary to avoid index errors
    #  *** If you create a model on a small vocab and want a smaller embedding size, remove this test ***
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    ## Preprocess Dataset
    ## Tokenize all the texts
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a 'model_max_length' that is longer than the default 'block_size' value (1024)"
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length ({args.max_seq_length}) is larger than the maximum length for"
                f"the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        ## Tokenize each non-empty line
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            ## Remove empty line
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,  # with 'special_tokens_mask', DataCollatorForLanguageModeling is more efficient
            )

        with accelerator.main_process_first():
            if not args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocess_num_worker,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        ## Tokenize every text -> Concatenate together before splitting them in smaller parts
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name],
                return_special_tokens_mask=True,  # with 'special_tokens_mask', DataCollatorForLanguageModeling is more efficient
            )

        with accelerator.main_process_first():
            if not args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocess_num_worker,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        def group_texts(examples):
            ## Concatenate all texts
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            ## Split by chunks of max_len
            result = {
                k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        with accelerator.main_process_first():
            if not args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=args.preprocess_num_worker,
                    load_from_cache_file=not args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    ## Data Collator -> take care of randomly masking the tokens
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    ## DataLoaders creation
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size
    )

    ## Optimizer
    ## Split weights in 2 Groups : i) with weight decay, ii) without weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    ## Manage with argparse
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    ## Calculate the number of Train steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    ## Scheduler
    lr_scheduler = get_scheduler(
        name=args.sche,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.grad_accum_steps,
        num_training_steps=args.max_train_steps * args.grad_accum_steps,
    )

    ## Prepare with Accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    ## Recalculate total Train steps (as the size of Train DataLoader may have changed)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.train_epoch * num_update_steps_per_epoch
    args.train_epoch = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_train", experiment_config)

    if args.log_wandb:
        if accelerator.is_main_process:
            wandb.init(project=args.project, name=args.run_name, config=args, reinit=True)

    # *** Start Train! *** #
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Length of DataLoader = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.grad_accum_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    starting_epoch = 0

    ## Load weights & states from Checkpoint
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)

        accelerator.print(f"Resumed from Checkpoint : {checkpoint_path}")
        accelerator.load_state(checkpoint_path)

        ## Extract 'epoch_{i}' or 'step_{i}'
        training_difference = os.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    ## Train loop
    for epoch in range(starting_epoch, args.train_epoch):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        if accelerator.is_main_process:
            print("Train Progress")
        train_progress_bar = tqdm(range(len(active_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
            train_progress_bar.update(1)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.out_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        if accelerator.is_main_process:
            print("Evaluation Progress")
        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.eval_batch_size)))
            eval_progress_bar.update(1)

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "epoch": epoch,
                    "step": completed_steps,
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                },
                step=completed_steps,
            )

        if args.log_wandb:
            ## log train & eval metrics to Wandb
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": completed_steps,
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss.item() / len(train_dataloader),
                    }
                )

        if args.checkpointing_steps == "epoch":
            ## Save checkpoint for every epoch
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.out_dir, output_dir)
            accelerator.save_state(output_dir)

            ## Remove the oldest checkpoint if len(checkpoint_files) > num_checkpoint_hist

    accelerator.end_training()

    if args.out_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(args.out_dir, "model"), is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(os.path.join(args.out_dir, "tokenizer"))
            with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

    wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    model_config = set_config(args)
    main(model_config, args)
