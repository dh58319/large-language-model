# Tutorial for Pretraining LLM
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
  - Prepare model to use from [Huggingface Hub-Model](https://huggingface.co/models)
    - ex) Pretrain '[bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)' (sort: most likes)\
      What you have to remember is
      ```bash
      prajjwal1/bert-tiny
      ```
  - Prepare dataset to use from [Huggingface Hub-Dataset](https://huggingface.co/datasets)
    - ex) for '[wikipedia](https://huggingface.co/datasets/wikipedia)'\
      What you have to remember is
      ```bash
      wikipedia
      ```
      If you need pre-processed subsets(ex. 20220301.en),
      ```bash
      wikipedia, 20220301.en
      ```
### Step 3️⃣ Import model & configuration
```bash
# *** for example *** #

# import model & configuration you build
from model.BERT.modeling_bert import BertForMaskedLM as scratch_model
from model.BERT.configuration_bert import BertConfig

# Set configuration 
def set_config(args):
    return BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, attention_probs_dropout_prob=args.drop_prob)
```

### Step 4️⃣ Start train
```bash
# *** for example *** #
["CUDA_VISIBLE_DEVICES"]='0,' \         # no. of GPU to use
accelerate \ 
run_mlm.py \                            # run_script
--dataset wikipedia \                   # from step 2️⃣
--dataset_config 20220301.en \          # from step 2️⃣
--data_dir ~/shared/... \                   
--streaming \                           # True : not use local disk / False : store dataset in local disk
--tokenizer prajjwal1/bert-tiny \       # from step 2️⃣
...                                     # other arguments...
```
