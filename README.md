# Large Language Model
Let's learn about how to Pretrain or Fine-tune Large-Language-Model with Pytorch!\
First of all, all script is referred to [🤗Huggingface.transformers](https://github.com/huggingface/transformers/tree/main).\
Before training LLM, I recommend you to read the paper about the target model first.

## Model Architecture List
|                                                      Model                                                      |            Huggingface-Hub URL            | Model Dir  | 
|:---------------------------------------------------------------------------------------------------------------:|:-----------------------------------------:|:----------:|
|                                 [BERT](https://arxiv.org/pdf/1810.04805v2.pdf)                                  | https://huggingface.co/bert-base-uncased  |     -      |
|                              [DistilBERT](https://arxiv.org/pdf/1910.01108v4.pdf)                               |                     -                     |     -      |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  |                     -                     |     -      |
|                                [RoBERTa](https://arxiv.org/pdf/1907.11692v1.pdf)                                |                     -                     |     -      |


## Dataset List
### Pretrain
|    Data     |             Huggingface-Hub URL             |                      Data Dir                      |
|:-----------:|:-------------------------------------------:|:--------------------------------------------------:|
|  Wikipedia  |  https://huggingface.co/datasets/wikipedia  |  ~/shared/hdd_ext/nvme1/public/language/wikipedia  |
|  Wikitext   |  https://huggingface.co/datasets/wikitext   |  ~/shared/hdd_ext/nvme1/public/language/wikitext   |
### Fine-tune
| Data | Data Dir                                    |
|------|---------------------------------------------|
| GLUE | ~/shared/hdd_ext/nvme1/public/language/glue |

## Model Zoo
### Pretrain
- Wikipedia
  - epoch : 40

  |    Model    |  L/H  | perplexity | eval_loss |                                              model                                             | 
  |:-----------:|:-----:|:----------:|:---------:|:----------------------------------------------------------------------------------------------:|
  |  bert-tiny  | 2/128 |   26.211   |   3.266   | [download](https://drive.google.com/file/d/1R7VYGkFPa41dMzbnEla1TJWBFrYnAU-Y/view?usp=sharing) |
  |  bert-mini  | 4/256 |   8.057    |   2.087   | [download](https://drive.google.com/file/d/1S9GuJG7IPI0ogmXhmbkFqJ8cdFCsY_pJ/view?usp=sharing) |

### Fine-tune
- GLUE
  - with Pretrained model from above 'Pretrain' table
  - Batch size : 8, 16, 32, 64, 128
  - Learning rate : 1e-4, 3e-4, 3e-5, 5e-5
  - epoch : 4

  |   Model    | CoLA  |  SST-2  |     MRPC      |     QQP     |    STS-B    | MNLI  | QNLI  | WNLI  |  RTE  |                                            model                                               |
  |:----------:|:-----:|:-------:|:-------------:|:-----------:|:-----------:|:-----:|:-----:|:-----:|:-----:|:----------------------------------------------------------------------------------------------:|
  | bert-tiny  | 17.39 |  83.03  |  82.83/73.77  | 65.71/77.47 | 40.85/40.28 | 57.31 | 64.29 | 59.15 | 58.84 | [download](https://drive.google.com/file/d/1RyRXSx_9Rew2BTtUPigPhv3SHhMp6PUa/view?usp=sharing) |
  | bert-mini  | 26.97 |  87.27  |  87.92/83.09  | 81.00/85.74 | 84.19/84.44 | 69.06 | 84.26 | 57.75 | 62.09 | [download](https://drive.google.com/file/d/1UJU6vSTPDF67w9ueMfwm4gGUuJl-QYzU/view?usp=sharing) |

# End
More contents and more scripts will be added.\
Waiting for your feedback!
