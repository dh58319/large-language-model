# Large Language Model
Let's learn about how to 🔴 Pretrain or 🔵 Fine-tune Large-Language-Model with Pytorch!\
First of all, all script is referred to [🤗Huggingface.transformers](https://github.com/huggingface/transformers/tree/main).\
Before training LLM, I recommend you to read the paper about the target model first.\

## Model Architecture List
| Model | Huggingface-Hub URL | Model Dir | 
|-|-|-|
| [BERT](https://arxiv.org/pdf/1810.04805v2.pdf) | https://huggingface.co/bert-base-uncased |-|
| [DistilBERT]() |-|-|

## Dataset List
| Data | Huggingface-Hub URL | Data Dir |
|-|-|-|
| Wikipedia| https://huggingface.co/datasets/wikipedia | ~/shared/hdd_ext/nvme1/public/language/wikipedia |
| Wikitext-raw | https://huggingface.co/datasets/wikitext | ~/shared/hdd_ext/nvme1/public/language/wikitext |

# Tutorial for Training LLM
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
### Step 3️⃣ Option) Import Custom Model
  If you want to import the Custom model(ex. [BERT](https://arxiv.org/pdf/1810.04805v2.pdf)) in 'run_mlm.py', add
  ```bash
  from BERT.bert_scratch import BertForMaskedLM as scratch_model
  ```
  ⚠️ If you want to load pretrained Model, Skip this step!!

### Step 4️⃣ Option) Set Configuration
  





# End
More contents and more scripts will be added.\
Is there anything what you want?
Waiting for your feedback!
