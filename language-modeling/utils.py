import logging
import os
from datetime import timedelta

import datasets
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

import transformers

def sanity_check(logger, args):
    if args.dataset is None and args.train_file is None and args.valid_file is None:
        raise ValueError("Need either 'args.dataset' or 'args.train_file' or 'args.valid_file'.")

    if args.dataset is not None and args.data_dir is None:
        logger.warning(
            "Load dataset from initial cache dir or download from huggingface hub."
            "Need 'args.data_dir' to load stored dataset."
        )
    else:
        logger.warning(
            "Load stored dataset(args.train_file, args.valid_file)."
        )
        if args.train_file:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl"]:
                raise ValueError("'args.train_file' should be a csv, json(l) or txt file.")
        if args.valid_file:
            extension = args.valid_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl"]:
                raise ValueError("'args.valid_file' should be a csv, json(l) or txt file.")
def init_accelerator(args):
    ## Initialize the accelerator
    accelerate_log_kwargs = {}
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps, kwargs_handlers=[kwargs], **accelerate_log_kwargs)
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
        raw_datasets = load_dataset(args.dataset, args.dataset_config, cache_dir=args.data_dir,
                                    streaming=args.streaming, trust_remote_code=True)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset,
                args.dataset_config,
                cache_dir=args.data_dir,
                split=f"train[:{args.valid_split_percentage}%]",
                streaming=args.streaming,
                trust_remote_code=True
            )
            raw_datasets["train"] = load_dataset(
                args.dataset,
                args.dataset_config,
                cache_dir=args.data_dir,
                split=f"train[{args.valid_split_percentage}%:]",
                streaming=args.streaming,
                trust_remote_code=True
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.valid_file:
            data_files["valid"] = args.valid_file
            extension = args.valid_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

        if "validation" not in raw_datasets.keys():
            # No 'args.valid_file', use 'args.valid_split_percentage' to divide the dataset
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets

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

