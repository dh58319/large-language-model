# reference : https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py

import argparse
import json
import math
import os

import wandb
import evaluate
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    get_scheduler,
)

from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from .utils import init_accelerator, make_log, load_checkpoint_utils

from ..model.bert.Bert import BertSequenceClassification as ModelForSequenceClassification

## Error will be occured if minimal version of Transformers is not installed
check_min_version("4.38.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sweep_config = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "grad_accum_steps": {"values": [1, 2, 4, 8, 16]},
        "lr": {"values": [1e-4, 3e-4, 3e-5, 5e-5]},
    },
}

parser = argparse.ArgumentParser(description="Finetuning on GLUE - Pytorch Large Language Model")

group = parser.add_argument_group('Dataset')
group.add_argument('--task', type=str, default=None, help="Name of the GLUE task to train on",
                   choices=list(task_to_keys.keys()))
group.add_argument('--cache_dir', type=str, help="Directory of stored Custom Dataset")
group.add_argument('-s', '--streaming', action='store_true', default=False,
                   help="Enable streaming mode -> not require local disk usage")

group = parser.add_argument_group('Model')
group.add_argument('--model', type=str,
                   help="Name or Path of the Pretrained model")

group = parser.add_argument_group('Parameter')
group.add_argument('--max_length', type=int, default=128,
                   help=(
                       "Maximum total input sequence length after tokenization"
                       "Sequences longer than this will be truncated"
                       "Sequences shorter than this will be padded if '--pad_to_max_length' is passed"
                   ))
group.add_argument('--pad_to_max_length', action="store_true",
                   help="Pad all samples to 'max_length'. Otherwise, dynamic padding is used.")
group.add_argument('--train_batch_size', type=int, default=8, help="Batch size per device for Training dataloader")
group.add_argument('--eval_batch_size', type=int, default=8, help="Batch size per device for Evaluation dataloader")
group.add_argument('--lr', type=float, default=5e-5,
                   help="Initial Learning Rate to use (after the potential warmup period)")
group.add_argument('--sche', type=str, default="linear", help="Learning Rate scheduler type")
group.add_argument('--weight_decay', type=float, default=0.0, help="Weight Decay to use")
group.add_argument('--train_epoch', type=int, default=3, help="Total number of training Epochs")
group.add_argument('--max_train_steps', type=int, default=None,
                   help=(
                       "Total number of training Steps"
                       "If provided, overrides num_train_epochs"
                   ))
group.add_argument('--num_warmup_steps', type=int, default=0, help="Number of steps for the Warmup in LR scheduler")
group.add_argument('--grad_accum_steps', type=int, default=1,
                   help="Number of update steps to accumulate before performing a backward/update pass")

group = parser.add_argument_group('Setting')
group.add_argument('--log_wandb', action='store_true', default=False,
                   help="log training and validation metrics to wandb")
group.add_argument('--project', type=str, default='',
                   help="name of train project, name of sub-folder for output")
group.add_argument('--run_name', type=str, default=False, help="run name for wandb")
group.add_argument('--sweep', action='store_true', default=False, help="Use Sweep for tuning hyperparameters")
group.add_argument('--out_dir', type=str, default=None, help="Directory to store the final model")
group.add_argument('--checkpointing_steps', type=str, default=None,
                   help=(
                       " default: not save checkpoint "
                       " 'n'(ex.'10'): save checkpoint for every n(ex.10) step "
                       " 'epoch': save checkpoint for every epoch "
                   ))
group.add_argument('--resume_from_checkpoint', type=str, default=None,
                   help="Directory to continue training from Checkpoint")

parser.add_argument('--seed', type=int, default=None, help="Seed for Reproducible training")
parser.add_argument('--trust_remote_code', action="store_true", default=False,
                    help=(
                        "Allow custom models (defined on the Hub - https://huggingface.co/models)"
                        "true : Trust repositories & Execute code present on the Hub"
                    ))
parser.add_argument('--with_tracking', action="store_true", default=False,
                    help="Enable experiment trackers for Logging")
parser.add_argument('--ignore_mismatched_sizes', action="store_true",
                    help="Enable to load Pretrained model whose head dimensions are different")

def main(args):
    ## Initialize the accelerator
    accelerator = init_accelerator(args)

    for grad_accum_step, lr in \
            zip(sweep_config["parameters"]["grad_accum_steps"]["values"], sweep_config["parameters"]["lr"]["values"]):
        args.grad_accum_steps = grad_accum_step
        args.lr = lr

    ## start wandb & sweep(optional)
    if args.log_wandb:
        if accelerator.is_main_process:
            wandb.init(project=f"{args.project}_{args.task}", name=args.run_name)

        if args.sweep:
            args.grad_accum_steps = wandb.config.grad_accum_steps
            args.lr = wandb.config.lr

    ## Make one log on every process with the configuration for debugging.
    make_log(logger, accelerator)

    ## Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    ## Create Output directory if it does not exist
    if accelerator.is_main_process:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ## Get Dataset
    ## public datasets are available on the hub - https://huggingface.co/datasets/
    raw_datasets = load_dataset("glue", args.task, cache_dir=args.cache_dir, streaming=args.streaming, trust_remote_code=True)
    ## Labels
    is_regression = args.task == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    ## Load Pretrained Model & Tokenizer
    config = AutoConfig.from_pretrained(
        args.model, num_labels=num_labels, finetuning_task=args.task, trust_remote_code=args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = ModelForSequenceClassification.from_pretrained(
        args.model, config=config, ignore_mismatched_sizes=args.ignore_mismatched_sizes, trust_remote_code=args.trust_remote_code
    )

    ## Preprocess Dataset
    sentence1_key, sentence2_key = task_to_keys[args.task]

    ## Some model have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}."
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset:"
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result."
            )
    elif args.task is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        ## Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                ## Map labels to ID (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                ## In all cases, rename the column to labels because the model will expect that
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        if not args.streaming:
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset"
            )
        else:
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
            )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task == "mnli" else "validation"]

    ## Dataloaders
    if args.pad_to_max_length:
        ## Convert everything to tensors
        data_collator = default_data_collator
    else:
        ## Apply dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

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
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    ## Prepare with 'accelerator'
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    ## Recalculate total training steps as the size of train_dataloader may have changed
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
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    ## Metric function
    if args.task:
        metric = evaluate.load("glue", args.task)
    else:
        metric = evaluate.load("accuracy")

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
    best_metric = {}

    ## Load weights & states from Checkpoint
    if args.resume_from_checkpoint:
        resume_step, completed_steps, starting_epoch =\
            load_checkpoint_utils(args, accelerator, train_dataloader, num_update_steps_per_epoch)

    for epoch in range(starting_epoch, args.train_epoch):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        if accelerator.is_main_process:
            print("Train Process")
        train_progress_bar = tqdm(range(len(active_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(active_dataloader):
            outputs = model(batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.out_dir:
                        output_dir = os.path.join(args.out_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        if accelerator.is_main_process:
            print("Evaluation Progress")
        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))

            if accelerator.num_processes > 1:
                ## in a multiprocess environment, the last batch has duplicates
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            eval_progress_bar.update(1)

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        if epoch == 0 or best_metric[list(best_metric.keys())[0]] < eval_metric[list(eval_metric.keys())[0]]:
            best_metric = eval_metric
            ## Save Best eval_metric model
            if args.out_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    os.path.join(args.out_dir, args.run_name), is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(os.path.join(args.out_dir, args.run_name))

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task is not None else "glue": eval_metric,
                    "best_accuracy" if args.task is not None else "best_glue": best_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.log_wandb:
            ## log train & eval metrics to Wandb
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "accuracy" if args.task is not None else "glue": eval_metric,
                        "best_accuracy" if args.task is not None else "glue": best_metric,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    }
                )

        if args.checkpointing_steps == "epoch":
            ## Save checkpoint for every epoch
            output_dir = f"epoch_{epoch}"
            if args.out_dir is not None:
                output_dir = os.path.join(args.out_dir, output_dir)
            accelerator.save_state(output_dir)

    accelerator.end_training()

    eval_metric = best_metric

    if args.task == "mnli":
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.out_dir is not None:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.out_dir, args.run_name, f"results_{args.run_name}.json"), "w") as f:
            json.dump(all_results, f)

    if args.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.sweep:
        sweep_id = wandb.sweep(sweep=sweep_config, project=f"{args.project}_{args.task}")
        wandb.agent(sweep_id, function=main(args), count=1)
    else:
        main(args)

