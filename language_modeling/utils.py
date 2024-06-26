import logging
import os
from datetime import timedelta
import numpy as np
import random

import torch
import argparse

import datasets
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

import transformers

def parse_argument():
    parser = argparse.ArgumentParser(
        description="Pretraining BERT")

    group = parser.add_argument_group('Dataset')
    ## Download or Stream dataset from Huggingface hub
    group.add_argument('--dataset', type=str, help="The name of the dataset")
    group.add_argument('--dataset_config', type=str, help="The configuration name of the dataset")
    group.add_argument('--cache_dir', type=str, default=None, help="Directory of stored Dataset")
    group.add_argument('-s', '--streaming', action='store_true', default=False,
                       help="Enable streaming mode -> not require local disk usage")

    ## Load Stored dataset
    group.add_argument('--tokenizer', type=str,
                       help="Name or Path of the Pretrained tokenizer (required if model_name != tokenizer_name)")
    group.add_argument('--load_raw_data', action='store_true', default=False,
                       help="Load raw dataset files, not preprocessed dataset files")
    group.add_argument('--save_tokenized_dataset', action='store_true', default=False,
                       help="Save tokenized datasets after preprocessing and tokenizing the raw datasets")
    group.add_argument('--extension', type=str, default='arrow',
                       help="Extension of files(.txt, .json, .arrow...)")
    group.add_argument('--train_file', type=str,
                       help=("File name which contains the train data -> extension : .csv, .json, .txt, .arrow"
                             "Path which contains the multiple train data"
                             ))
    group.add_argument('--valid_file', type=str,
                       help=("File name contains the validation data -> extension : .csv, .json, .txt, .arrow"
                             "Path which contains the multiple validation data"
                             ))
    group.add_argument('--valid_split_percentage', type=int, default=5,
                       help="Percentage of Train set used as Validation set if there is no Valid split")

    group.add_argument('--pad_to_max_length', action='store_true', default=False,  ### MLM ###
                       help="Pad all samples to 'max_length'. Otherwise, dynamic padding is used")
    group.add_argument('--max_seq_length', type=int, default=None,  ### MLM ###
                       help=(
                           "Maximum total input sequence length after tokenization."
                           "Sequences longer than this value will be truncated."
                       ))
    group.add_argument('--line_by_line', action='store_true', default=False,  ### MLM ###
                       help="Distinguish lines of text in Dataset -> Distinct Sequences")

    group = parser.add_argument_group('Parameter')
    group.add_argument('--train_batch_size', type=int, default=16,
                       help="Batch size for Training dataloader (per device)")
    group.add_argument('--eval_batch_size', type=int, default=16,
                       help="Batch size for evaluation dataloader (per device)")
    group.add_argument('--mlm_probability', type=float, default=0.15,
                       help="Probability of masked tokens for MLM")  ### MLM ###
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
    group.add_argument('--drop_prob', type=float, default=0.0, help="Dropout Probability")  ### MLM ###

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
    parser.add_argument('--preprocess_num_worker', type=int, default=None,
                        help="Number of process for Preprocessing")
    parser.add_argument('--overwrite_cache', action="store_true", help="Overwrite the cached Train & Eval sets")
    parser.add_argument('--trust_remote_code', action="store_true", default=False,
                        help=(
                            "Allow custom models (defined on the Hub - https://huggingface.co/models)"
                            "true : Trust repositories & Execute code present on the Hub"
                        ))
    parser.add_argument('--low_cpu_mem_usage', action="store_true",
                        help=(
                            "Create the model as an empty shell -> only materialize parameters when pretrained models are loaded"
                            "If passed, LLM loading time and RAM comsumption will be benefited"
                        ))
    return parser

def sanity_check(accelerator, args):
    if args.dataset is None and\
            args.train_file is None and args.valid_file is None:
        raise ValueError("Need either 'args.dataset' or 'args.train_file' or 'args.valid_file'.")

    if args.dataset is not None and args.data_dir is None:
        if accelerator.is_main_process:
            print(
                "Load dataset from initial cache dir or download from huggingface hub."
                "Need 'args.data_dir' to load stored dataset."
            )
    else:
        if accelerator.is_main_process:
            print(
                "Load stored dataset(args.train_file, args.valid_file)."
            )
        if args.train_file:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl", "arrow"]:
                raise ValueError("'args.train_file' should be a csv, json(l), arrow or txt file.")
        if args.valid_file:
            extension = args.valid_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl", "arrow"]:
                raise ValueError("'args.valid_file' should be a csv, json(l), arrow or txt file.")

def init_accelerator(args):
    ## Initialize the accelerator
    accelerate_log_kwargs = {}
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800000))
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.grad_accum_steps, kwargs_handlers=[kwargs], **accelerate_log_kwargs)
    return accelerator

def make_log(logger, accelerator):
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

def load_dataset_utils(args):
    ## public datasets are available on the hub - https://huggingface.co/datasets/
    if args.dataset:
        ## Load dataset from Huggingface hub
        datasets = load_dataset(args.dataset, args.dataset_config, cache_dir=args.cache_dir,
                                    streaming=args.streaming, trust_remote_code=True)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                args.dataset,
                args.dataset_config,
                cache_dir=args.cache_dir,
                split=f"train[:{args.valid_split_percentage}%]",
                streaming=args.streaming,
                trust_remote_code=True
            )
            datasets["train"] = load_dataset(
                args.dataset,
                args.dataset_config,
                cache_dir=args.cache_dir,
                split=f"train[{args.valid_split_percentage}%:]",
                streaming=args.streaming,
                trust_remote_code=True
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.valid_file:
            data_files["validation"] = args.valid_file
        if args.extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        if args.extension == "jsonl" or "json":
            extension = "json"
        if args.extension == "arrow":
            extension = "arrow"
        else:
            ValueError("Need args.extension to load stored dataset")

        if os.path.isdir(args.train_file):
            datasets = {}
            for task, path in zip(["train", "validation"], [args.train_file, args.valid_file]):
                all_files = os.listdir(path)
                data_files = [file for file in all_files if file.split('.')[-1] == args.extension]
                datasets[task] = concatenate_datasets([Dataset.from_file(os.path.join(path, data_file)) for data_file in data_files])
            datasets = DatasetDict(datasets)
        else:
            datasets = load_dataset(extension, data_files=data_files, **dataset_args)

        if "validation" not in datasets.keys():
            # No 'args.valid_file' -> use 'args.valid_split_percentage' to divide the dataset
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.valid_split_percentage}%]",
                **dataset_args,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.valid_split_percentage}%:]",
                **dataset_args,
            )
    return datasets

def load_checkpoint_utils(args, accelerator, train_dataloader, num_update_steps_per_epoch):
    ## Load weights & states from Checkpoint
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
        resume_step = int(training_difference.replace("step_", "")) * args.grad_accum_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.grad_accum_steps
        resume_step -= starting_epoch * len(train_dataloader)

    return resume_step, completed_steps, starting_epoch

def jsonl_to_json(jsonl_file_path):
    import json
    import jsonlines

    json_file_path = str(jsonl_file_path.rsplit('.', 1)[0]) + '.json'
    if os.path.exists(json_file_path):
        return json_file_path, "json"

    json_file = {}
    with jsonlines.open(jsonl_file_path) as f:
        for line_idx, line in enumerate(f):
            key_list = list(line.keys())
            value_list = list(line.values())
            for idx, key in enumerate(key_list):
                if line_idx == 0:
                    json_file[key] = []
                json_file[key].append(value_list[idx])

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_file, f)

    return json_file_path, "json"

def is_next_labeling(text_data, sentence1, sentence2):
    if random.random() > 0.5:
        return torch.LongTensor([1]), [sentence1, sentence2]
    else:
        rand_article = random.randint(0, len(text_data) - 1)
        rand_line = random.randint(0, len(text_data[rand_article]) - 1)
        return torch.LongTensor([0]), [sentence1, text_data[rand_article][rand_line]]

def random_masking(sentence, mask_ratio=0.15, tokenizer=None):
    words = np.array(sentence.split())

    output, output_label = [], []
    for word in words:
        prob = random.random()
        ## Replace tokens
        if prob < mask_ratio:
            prob /= mask_ratio
            if prob < 0.8:
                output.append('[MASK]')
            elif prob < 0.9:
                output.append(random.choice(list(tokenizer.vocab.keys())))
            else:
                output.append(word)
            output_label.append(word)
        else:
            output.append(word)
            output_label.append('[PAD]')

    return output, output_label

def add_cls_sep(sentence1, sentence2, max_seq_length, tokenizer=None):
    output = (sentence1 + [tokenizer.vocab['[SEP]']]
              + sentence2 + [tokenizer.vocab['[SEP]']])[:max_seq_length]
    segment_label = ([1 for _ in range(len(sentence1))] + [2 for _ in range(len(sentence2))])[:max_seq_length]
    return output, segment_label

def preprocess_function(examples, text_column_name="text"):
    text_data = []
    for i in range(len(examples[text_column_name])):
        lines = examples[text_column_name][i].replace('\t', '\n').replace('. ', '.\n').split('\n')
        text_data.append([line for line in lines if len(line) > 0 and not line.isspace()])
    text_data = [line for line in text_data if len(line) > 0]

    is_next_labels, sentences_pairs = [], []
    for idx, article in enumerate(text_data):
        lines = np.array(article)
        for line1, line2 in zip(lines[:-1], lines[1:]):
            is_next_label, sentences_pair = is_next_labeling(text_data, line1, line2)
            is_next_labels.append(is_next_label)
            sentences_pairs.append(sentences_pair)

    return {"is_next_labels": is_next_labels,
            text_column_name: sentences_pairs,
            }

if __name__ == "__main__":
    pass