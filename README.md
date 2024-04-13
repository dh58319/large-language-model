# Large Language Model
Let's learn about how to Pretrain or Fine-tune Large-Language-Model with Pytorch!\
First of all, all script is referred to [🤗Huggingface.transformers](https://github.com/huggingface/transformers/tree/main).\
Before training LLM, I recommend you to read the paper about the target model first.

## Model Architecture List
|                                                     Model                                                     |                     Huggingface-Hub URL                     | Model Dir | 
|:-------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------:|:---------:|
|                                [BERT](https://arxiv.org/pdf/1810.04805v2.pdf)                                 |   https://huggingface.co/models?sort=trending&search=bert   |     -     |
| [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |   https://huggingface.co/models?sort=trending&search=gpt2   |     -     |
|                                 [T5](https://arxiv.org/pdf/1910.10683v4.pdf)                                  |    https://huggingface.co/models?sort=trending&search=t5    |     -     | 
|                               [RoBERTa](https://arxiv.org/pdf/1907.11692v1.pdf)                               |                              -                              |     -     |
|                             [DistilBERT](https://arxiv.org/pdf/1910.01108v4.pdf)                              |                              -                              |     -     |



## Dataset List
### for Pretrain
|    Data     |                    Reference URL                     |            ( Raw / Tokenized ) Data Dir             | 
|:-----------:|:----------------------------------------------------:|:---------------------------------------------------:|
|  Wikipedia  |      https://huggingface.co/datasets/wikipedia       |   ~/public/language/wikipedia/20220301.en/2.0.0/    |
|  Wikitext   |       https://huggingface.co/datasets/wikitext       | ~/public/language/wikitext/wikitext-2-raw-v1/0.0.0/ |
|  MiniPile   | https://huggingface.co/datasets/JeanKaddour/minipile |   ~/public/language/minipile/default/0.0.0/         |
| OpenAI-GPT2 |    https://github.com/openai/gpt-2-output-dataset    |                          -                          |

### for Fine-tune
| Data | Data Dir                                    |
|------|---------------------------------------------|
| GLUE | ~/shared/hdd_ext/nvme1/public/language/glue |

## Model Zoo
### Pretrain
##### - Wikipedia - 
  - epoch : 40
  - Only Masked Language Model, No Next Sentence Prediction 

  |                          Model                          | Size | perplexity | eval_loss |                                             cfg                                              | 
  |:-------------------------------------------------------:|:----:|:----------:|:---------:|:--------------------------------------------------------------------------------------------:|
  | [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) |  4M  |   26.211   |   3.266   | [download](https://drive.google.com/uc?export=download&id=1R7VYGkFPa41dMzbnEla1TJWBFrYnAU-Y) |
  | [bert-mini](https://huggingface.co/prajjwal1/bert-mini) | 11M  |   8.057    |   2.087   | [download](https://drive.google.com/uc?export=download&id=1S9GuJG7IPI0ogmXhmbkFqJ8cdFCsY_pJ) |

##### - OpenAI-GPT2 -
  - This dataset is the alternative of WebText which is referred in the [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
  - OpenAI released this dataset for researchers. WebText is not released.
  - epoch : 40

|                        Model                         | Size | perplexity | eval_loss |                                             cfg                                              | 
|:----------------------------------------------------:|:----:|:----------:|:---------:|:--------------------------------------------------------------------------------------------:|
| [GPT2](https://huggingface.co/openai-community/gpt2) | 137M |  127.966   |   4.852   | [download](https://drive.google.com/uc?export=download&id=1UaK4CUaBhxbOwI_2ZlTTx6_Lzhp3Puia) |

### Fine-tune
##### - GLUE -
  - with Pretrained model from above 'Pretrain' table
  - Batch size : 8, 16, 32, 64, 128
  - Learning rate : 1e-4, 3e-4, 3e-5, 5e-5
  - epoch : 4

  |                          Model                           | CoLA  |  SST-2  |     MRPC      |     QQP     |    STS-B    | MNLI  | QNLI  | WNLI  |  RTE  |                                             cfg                                              |
  |:--------------------------------------------------------:|:-----:|:-------:|:-------------:|:-----------:|:-----------:|:-----:|:-----:|:-----:|:-----:|:--------------------------------------------------------------------------------------------:|
  | [bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)  | 17.39 |  83.03  |  82.83/73.77  | 65.71/77.47 | 40.85/40.28 | 57.31 | 64.29 | 59.15 | 58.84 | [download](https://drive.google.com/uc?export=download&id=1RyRXSx_9Rew2BTtUPigPhv3SHhMp6PUa) |
  | [bert-mini](https://huggingface.co/prajjwal1/bert-mini)  | 26.97 |  87.27  |  87.92/83.09  | 81.00/85.74 | 84.19/84.44 | 69.06 | 84.26 | 57.75 | 62.09 | [download](https://drive.google.com/uc?export=download&id=1UJU6vSTPDF67w9ueMfwm4gGUuJl-QYzU) |

# End
More contents and more scripts will be added.\
Waiting for your feedback!
