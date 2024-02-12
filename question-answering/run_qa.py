import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import wandb

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from utils import init_accelerator, make_log, load_checkpoint

## import model & model configuration
from transformers.models.bert.modeling_bert import BertForMaskedLM as scratch_model
from transformers.models.bert.configuration_bert import BertConfig

BERT_cfg = {
    ## prajjwal1/bert-##
    "prajjwal1/bert-tiny"  : [128, 2],
    "prajjwal1/bert-mini"  : [256, 4],
    "prajjwal1/bert-small" : [512, 4],
    "prajjwal1/bert-medium": [512, 8],
}

def set_config(args):
    return BertConfig(hidden_size=BERT_cfg[args.tokenizer][0], num_hidden_layers=BERT_cfg[args.tokenizer][1],
                      num_attention_heads=BERT_cfg[args.tokenizer][1], attention_probs_dropout_prob=args.drop_prob)

## Error will be occured if minimal version of Transformers is not installed
check_min_version("4.38.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

parser = argparse.ArgumentParser(description="Pretraining on Masked Language Modeling task - Pytorch Large Language Model")

group = parser.add_argument_group('Dataset')
group.add_argument('--dataset', type=str, help="The name of the dataset")
group.add_argument('--dataset_config', type=str, help="The configuration name of the dataset")
group.add_argument('--data_dir', type=str, default=None, help="Directory of stored Dataset")
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
group.add_argument('--max_seq_length', type=int, default=384,
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
group.add_argument('--grad_accum_steps', type=int, default=1,
                   help="Number of update steps to accumulate gradients before performing a backward/update pass")
group.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
group.add_argument('--mlm_probability', type=float, default=0.15, help="Probability of masked tokens for MLM")
group.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
group.add_argument('--train_epoch', type=int, default=3, help="Total number of Training epochs")
group.add_argument('--max_train_steps', type=int, default=None,
                   help="Total number of Training steps -> override 'epochs'")
group.add_argument('--sche', type=str, default="linear",
                   help="LR Scheduler : 'linear', 'cosine', cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'")
group.add_argument('--num_warmup_steps', type=int, default=0, help="Number of Warmup step in LR scheduler")
group.add_argument('--drop_prob', type=float, default=0.0, help="Dropout Probability")
group.add_argument('--doc_stride', type=int, default=128,
                   help="How much stride to take between chunks when split up a long document into chunks")
group.add_argument('--n_best_size', type=int, default=20,
                   help="Total number of n-best predictions to generate when looking for an answer")
group.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                   help=(
                       "Threshold used to select the null answer"
                   ))


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
                   ))
group.add_argument('--checkpointing_epochs', type=str, default=None,
                   help=(
                       " default: not save checkpoint "
                       " 'n'(ex.'10'): save checkpoint for every n(ex.10) epoch "
                   ))
group.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help="Directory to continue training from Checkpoint")

parser.add_argument('--seed', type=int, default=None, help="Seed for reproducible training")
parser.add_argument('--preprocess_num_worker', type=int, default=1,
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
    ## Initialize the accelerator
    accelerator = init_accelerator(args)

    if args.log_wandb:
        if accelerator.is_main_process:
            wandb.init(project=args.project, name=args.run_name, config=args, reinit=True)

    ## Make one log on every process with the configuration for debugging.
    make_log(logger, accelerator)

    ## Create Output directory if it does not exist
    if accelerator.is_main_process:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    ## Get Dataset
    raw_datasets = load_dataset(args.dataset, args.dataset_config, cache_dir=args.data_dir,
                                streaming=args.streaming, trust_remote_code=True)

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
        ## if model_name != tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)
    elif args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from Scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name"
        )

    if args.model:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        ## Train from Scratch -> Load Model
        model = scratch_model(config)
        logger.info("Training new model from Scratch")

    ## Preprocess Dataset
    column_names = raw_datasets["train"].column_names
    print(f"column : {column_names}")

    # question_column_name = "question" if "question" in column_names else


if __name__ == "__main__":
    args = parser.parse_args()
    model_config = set_config(args)

    main(model_config, args)