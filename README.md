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

|    Model    |  L/H  | perplexity | eval_loss |                                              cfg                                               | 
|:-----------:|:-----:|:----------:|:---------:|:----------------------------------------------------------------------------------------------:|
|  bert-tiny  | 2/128 |   26.211   |   3.266   | [download](https://drive.google.com/file/d/1R7VYGkFPa41dMzbnEla1TJWBFrYnAU-Y/view?usp=sharing) |
|  bert-mini  | 4/256 |   8.057    |   2.087   | [download](https://drive.google.com/file/d/1S9GuJG7IPI0ogmXhmbkFqJ8cdFCsY_pJ/view?usp=sharing) |

### Fine-tune
- GLUE
  - with Pretrained model from above 'Pretrain' table
  - Batch size : 8, 16, 32, 64, 128
  - Learning rate : 1e-4, 3e-4, 3e-5, 5e-5
  - epoch : 4

  |   Model    |   CoLA   |  MNLI   |     MRPC      | QNLI  |     QQP     |  RTE  | SST-2 |    STS-B    | WNLI  |     cfg      |
  |:----------:|:--------:|:-------:|:-------------:|:-----:|:-----------:|:-----:|:-----:|:-----------:|:-----:|:------------:|
  | bert-tiny  |  0.1739  |  57.31  |  82.83/73.77  | 64.29 | 65.71/77.47 | 58.84 | 83.03 | 40.85/40.28 | 59.15 | [download](https://drive.google.com/file/d/1RyRXSx_9Rew2BTtUPigPhv3SHhMp6PUa/view?usp=sharing) |
  | bert-mini  |  0.2697  |  69.06  |  87.92/83.09  | 84.26 | 81.00/85.74 | 62.09 | 87.27 | 84.19/84.44 | 57.75 | [download](https://drive.google.com/file/d/1UJU6vSTPDF67w9ueMfwm4gGUuJl-QYzU/view?usp=sharing) |

# End
More contents and more scripts will be added.\
Waiting for your feedback!
