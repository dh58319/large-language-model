# Pretrain LLM
In this section, I will introduce how to Pretrain LLM step by step.\
I try hard to explain each step as easily as I can.

### Step 1️⃣ Set environment
  - 'transformers >= 4.38.0.dev0' (if the lastest version is released, download that version)
  ```bash
  git clone https://github.com/huggingface/transformers
  cd transformers
  pip install .
  ```
  - cd in the folder where 'requirements.txt' is existed and install requirements
  ```bash
  pip install -r requirements.txt
  ```
### Step 2️⃣ Prepare model & dataset for training
  - Select model to use from [Huggingface Hub-Model](https://huggingface.co/models)
    - What you have to remember is
      ```bash
      prajjwal1/bert-tiny         ## from Hub
      ```
### Step 3️⃣ Import model & configuration
```bash
# *** for example *** #

# import model & configuration you build
from ..model.bert.Bert import BertLM as scratch_model
from transformers.models.bert.configuration_bert import BertConfig

# Set configuration 
# in script, the parameters are set with "args.tokenizer" automatically
# if you want to set those in person, just change like following lines
def set_config(args):
    return BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, attention_probs_dropout_prob=args.drop_prob)
```

### Step 4️⃣ Start train
- Read & Understand the script before start training, especially ArgumentParser!
```bash
# *** for example *** #
CUDA_DEVICE_ORDER='PCI_BUS_ID' \
CUDA_VISIBLE_DEVICES='0,' \                                    # no. of GPU to use
accelerate launch -m \                                         # Run Script as Module 
large-language-model.language-modeling.pretrain_bert \         # script for run
--tokenizer prajjwal1/bert-tiny \                              # from step 2️⃣
--train_file ~/public/language/minipile/default/0.0.0/tokenized_data/train \
--valid_file ~/public/language/minipile/default/0.0.0/tokenized_data/validation \
--extension arrow \                                            # extension of data files
--out_dir ~/directory/for/output \

~~~~~~ other arguments what you need ~~~~~~
...                         
```
