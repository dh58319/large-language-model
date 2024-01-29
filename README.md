# Large Language Model

## Introduction
hugging

## Model Architecture List
The location of Model architectures is [here]()


| Model | Huggingface-Hub URL |
|-|-|
| [BERT](https://arxiv.org/pdf/1810.04805v2.pdf) |-|

## Dataset


## Tutorial for Training LLM
In this section, I will introduce how to Pretrain or Fine-tune LLM models step by step.\
I try hard to explain the steps as easily as I can, so even beginners can understand and follow easily.

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
  Explore [Huggingface Hub](https://huggingface.co/docs/hub/index) to prepare model and dataset to use.
  - Prepare model to use from [Huggingface Hub-Model](https://huggingface.co/models)
    - ex) 🔴 Pretrain 'bert-tiny', search [bery-tiny](https://huggingface.co/prajjwal1/bert-tiny) (sort: most likes)
      What you have to prepare is
      ```bash
      prajjwal1/bert-tiny
      ```
    - ex) 🔵 Fine-tune 'bert-base-uncased', search [bert-base-uncased](https://huggingface.co/bert-base-uncased).
      What you have to prepare is
      ```bash
      bert-base-uncased
      ```
  - Prepare dataset to use from [Huggingface Hub-Dataset](https://huggingface.co/datasets)
    - ex) for 'Wikipedia', search [wikipedia](https://huggingface.co/datasets/wikipedia)
      What you have to prepare is
      ```bash
      wikipedia
      ```
      If you need pre-processed subsets(ex. 20220301.en),
      ```bash
      wikipedia, 20220301.en
      ```
### Step 3️⃣ Option) Import Custom Model & model Configuration
  If you want to import the Custom model(ex. [BERT](https://arxiv.org/pdf/1810.04805v2.pdf)) in 'run_mlm.py', add
  ```bash
  from BERT.bert_scratch import BertForMaskedLM as scratch_model
  ```

### Step 4️⃣ 
