# Fine-tune LLM
In this section, I will introduce how to Fine-tune LLM step by step.\
I try hard to explain each step as easily as I can.\
Introduced steps below are about fine-tuning pretrained models on GLUE task.

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
  - Select model to use from [Huggingface Hub-Model](https://huggingface.co/models) or saved cfg files
    - What you have to remember is
      ```bash
      bert-base-uncased                       ## from Hub
      # or
      /directory/containing/saved_cfg_files   ## from the directory
      ```
    - Select task name to train on
      - Choose one task(key) from
        ```bash
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        ```

### Step 3️⃣ Start train
```bash
CUDA_VISIBLE_DEVICES='0,' \         # no. of GPU to use
accelerate launch \ 
run_glue.py \                           # script for run
--task mnli \                           # from step 2️⃣
--data_dir ~/shared/... \
--model bert-base-uncased \             # from step 2️⃣
...                                     # other arguments... 
```