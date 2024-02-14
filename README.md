# Large Language Model
Let's learn about how to Pretrain or Fine-tune Large-Language-Model with Pytorch!\
First of all, all script is referred to [🤗Huggingface.transformers](https://github.com/huggingface/transformers/tree/main).\
Before training LLM, I recommend you to read the paper about the target model first.

## Model Architecture List
| Model | Huggingface-Hub URL | Model Dir | 
|-|-|-|
| [BERT](https://arxiv.org/pdf/1810.04805v2.pdf) | https://huggingface.co/bert-base-uncased |-|
| [DistilBERT](https://arxiv.org/pdf/1910.01108v4.pdf) |-|-|
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |-|-|

## Dataset List
### Pretrain
| Data      | Huggingface-Hub URL                       | Data Dir                                         |
|-----------|-------------------------------------------|--------------------------------------------------|
| Wikipedia | https://huggingface.co/datasets/wikipedia | ~/shared/hdd_ext/nvme1/public/language/wikipedia |
| Wikitext  | https://huggingface.co/datasets/wikitext  | ~/shared/hdd_ext/nvme1/public/language/wikitext  |
### Fine-tune
| Data | Data Dir                                    |
|------|---------------------------------------------|
| GLUE | ~/shared/hdd_ext/nvme1/public/language/glue |

## Model Zoo
### Pretrain
- Wikipedia

|    Model    |  epoch  |  perplexity  |  eval_loss  |                                               cfg                                                | 
|:-----------:|:-------:|:------------:|:-----------:|:------------------------------------------------------------------------------------------------:|
|  bert-tiny  |   40    |    26.211    |    3.266    | [download](https://drive.google.com/file/d/1R7VYGkFPa41dMzbnEla1TJWBFrYnAU-Y/view?usp=sharing) |
|  bert-mini  |   40    |    8.073     |    2.089    |                                                -                                                 |

### Fine-tune
- GLUE
  - Batch size : 8, 16, 32, 64, 128
  - Learning rate : 1e-4, 3e-4, 3e-5, 5e-5
  - epoch : 4

  |   Model    |   CoLA   |  MNLI   |      MRPC       | QNLI  |      QQP      |  RTE  | SST-2 | STS-B  | WNLI  | cfg  |
  |:----------:|:--------:|:-------:|:---------------:|:-----:|:-------------:|:-----:|:-----:|:------:|:-----:|:----:|
  | bert-tiny  |  0.1739  |  57.31  |  82.83 / 73.77  | 64.29 | 65.71 / 77.47 | 58.84 | 83.03 |   -    |   -   |  -   |
  | bert-mini  |    -     |    -    |        -        |   -   |       -       |   -   |   -   |   -    |   -   |  -   |

# End
More contents and more scripts will be added.\
Waiting for your feedback!
