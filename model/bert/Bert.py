import math

import os
import torch
import torch.nn as nn

from ..transformer.transformer import TransformerBlock
from transformers.models.bert import BertPreTrainedModel

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
        return x

class NextSentencePrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

        self.nsp_linear = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        pooled_output = self.tanh(self.dense(x[:, 0]))
        return self.nsp_linear(pooled_output)

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
        bert_output = self.bert(x)

        nsp_output = self.nsp(bert_output)
        mlm_output = self.mlm(bert_output)
        return nsp_output, mlm_output

if __name__ == "__main__":
    pass