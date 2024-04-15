## reference : https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py

import json
import math
import os

import wandb
import torch
import torch.nn as nn

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from .utils import parse_argument, init_accelerator, make_log, load_dataset_utils, load_checkpoint_utils

from ..model.bert.Bert import BertNoNSP as scratch_model
from ..model.bert.Bert_config import BertConfig

BERT_cfg = {
    # prajjwal1/bert-     [n_embd, n_layer]
    "prajjwal1/bert-tiny": [128, 2],
    "prajjwal1/bert-mini": [256, 4],
    "prajjwal1/bert-small": [512, 4],
    "prajjwal1/bert-medium": [512, 8],
}

def set_config(args):
    return BertConfig(hidden_size=BERT_cfg[args.tokenizer][0], num_hidden_layers=BERT_cfg[args.tokenizer][1],
                      num_attention_heads=BERT_cfg[args.tokenizer][1], attention_probs_dropout_prob=args.drop_prob)

## Error will be occured if minimal version of Transformers is not installed
check_min_version("4.38.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def main(model_config, args):
    ## Initialize the accelerator
    accelerator = init_accelerator(args)
    # sanity_check(accelerator, args)

    if args.log_wandb:
        if accelerator.is_main_process:
            wandb.init(project=args.project, name=args.run_name, config=args, reinit=True)

    ## Make one log on every process with the configuration for debugging.
    make_log(logger, accelerator)

    ## Set training seed
    if args.seed:
        set_seed(args.seed)

    ## Create Output directory if it does not exist
    if accelerator.is_main_process:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ## Load Pretrained Model & Tokenizer
    config = model_config
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    model = scratch_model(config)
    logger.info("Training new model from Scratch")

    if args.load_raw_data:
        ## Load raw data & preprocess for Next Sentence Prediction & tokenize the data
        datasets = load_dataset_utils(args)
        column_names = datasets["train"].column_names
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

        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )
        with accelerator.main_process_first():
            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocess_num_worker,
                remove_columns=[text_column_name],
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

        if args.save_tokenized_dataset:
            tokenized_datasets.save_to_disk(os.path.join(args.out_dir, "tokenized_data"))

    else:
        tokenized_datasets = load_dataset_utils(args)
        column_names = tokenized_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))
    loss_func = nn.CrossEntropyLoss()

    ## Calculate the number of Train steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.sche,
        optimizer=optimizer,
        # num_warmup_steps=args.num_warmup_steps * args.grad_accum_steps,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        # num_training_steps=args.max_train_steps * args.grad_accum_steps,
        num_training_steps=args.max_train_steps if overrode_max_train_steps
        else args.max_train_steps * accelerator,
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

    checkpointing_epochs = args.checkpointing_epochs
    if checkpointing_epochs is not None and checkpointing_epochs.isdigit():
        checkpointing_epochs = int(checkpointing_epochs)

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
        resume_step, completed_steps, starting_epoch = \
            load_checkpoint_utils(args, accelerator, train_dataloader, num_update_steps_per_epoch)

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
                outputs = model(batch)

                loss = loss_func(outputs.view(-1, config.vocab_size), batch["labels"].view(-1))
                total_loss += loss

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
                outputs = model(batch)

            loss = loss_func(outputs.view(-1, config.vocab_size), batch["labels"].view(-1))
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.eval_batch_size)))
            eval_progress_bar.update(1)

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.log_wandb:
            ## log train & eval metrics to Wandb
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": completed_steps,
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss / len(train_dataloader),
                    }
                )

        if isinstance(checkpointing_epochs, int):
            if epoch % checkpointing_epochs == 0:
                output_dir = f"epoch_{epoch}"
                if args.out_dir is not None:
                    output_dir = os.path.join(args.out_dir, output_dir)
                accelerator.save_state(output_dir)

        ## Save final state
        if epoch == args.train_epoch:
            output_dir = f"final_epoch_{epoch}"
            if args.out_dir is not None:
                output_dir = os.path.join(args.out_dir, output_dir)
            accelerator.save_state(output_dir)

        ## Remove the oldest checkpoint if len(checkpoint_files) > num_checkpoint_hist

    accelerator.end_training()

    if args.out_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(args.out_dir, args.run_name), is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(os.path.join(args.out_dir, args.run_name))
            with open(os.path.join(args.out_dir, f"all_results_{args.run_name}.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)

    if args.log_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = parse_argument()
    args = parser.parse_args()
    model_config = set_config(args)
    main(model_config, args)
