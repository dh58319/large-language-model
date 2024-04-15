import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ..transformer.transformer import TransformerBlock
from transformers.models.bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, x):
        input_shape = x["input_ids"].size()
        seq_len = input_shape[1]
        position_ids = self.position_ids[:, 0: seq_len]

        input_embeddings = self.word_embeddings(x["input_ids"])
        token_type_embeddings = self.token_type_embeddings(x["token_type_ids"])
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + token_type_embeddings + position_embeddings

        return self.dropout(self.layer_norm(embeddings))

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding = BertEmbedding(config)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.transformer_blocks[layer].attention.prune_heads(heads)

    def forward(self, x):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        pooled_output = self.tanh(self.dense(x[:, 0]))
        return x, pooled_output

class NextSentencePrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nsp_linear = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        return self.nsp_linear(x)

class MaskedLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlm_linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.mlm_linear.bias = self.bias

    def forward(self, x):
        hidden_states = self.layer_norm(self.gelu(self.dense(x)))
        return self.mlm_linear(hidden_states)

class BertNoNSP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.mlm = MaskedLanguageModel(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.mlm.mlm_linear

    def set_output_embeddings(self, new_embeddings):
        self.mlm.mlm_linear = new_embeddings

    def forward(self, x):
        bert_output = self.bert(x)

        mlm_output = self.mlm(bert_output)
        return mlm_output

class BertLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.nsp = NextSentencePrediction(config)
        self.mlm = MaskedLanguageModel(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.mlm.mlm_linear

    def set_output_embeddings(self, new_embeddings):
        self.mlm.mlm_linear = new_embeddings

    def forward(self, x):
        bert_output, pooled_output = self.bert(x)

        nsp_output = self.nsp(pooled_output)
        mlm_output = self.mlm(bert_output)
        return nsp_output, mlm_output

# for finetuning on GLUE
# (reference : https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
class BertSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

        self.post_init()

    def forward(self, x):
        outputs = self.bert(x)
        pooled_output = self.tanh(self.dense(outputs[:, 0]))

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        labels = x["labels"]
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    pass