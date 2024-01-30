# Tutorial for Training LLM
In this section, I will introduce how to Pretrain or Fine-tune LLM step by step.\
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
  - Prepare model to use from [Huggingface Hub-Model](https://huggingface.co/models)
    - ex) Fine-tune pretrained model '[bert-base-uncased](https://huggingface.co/bert-base-uncased)'\
      What you have to prepare is
      ```bash
      bert-base-uncased
      ```
  - Prepare dataset to use from [Huggingface Hub-Dataset](https://huggingface.co/datasets)
    - ex) for '[wikipedia](https://huggingface.co/datasets/wikipedia)'\
      What you have to prepare is
      ```bash
      wikipedia
      ```
      If you need pre-processed subsets(ex. 20220301.en),
      ```bash
      wikipedia, 20220301.en
      ```
### Step 3?? Move onto the branch you want and continue Tutorial from README in that branch
- ? Pretrain
  - MLM(MaskedLanguageModel)
- ? Fine-tune
  - GLUE
  - 
