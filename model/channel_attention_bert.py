import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.bert import BertPreTrainedModel

from .bert.Bert import BertEmbedding, MaskedLanguageModel
from .transformer.transformer import TransformerBlock, Attention

class ChannelAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x) # b, l, d

        q_layer = self.transpose_for_scores(q)  # b, h, l, d/h
        k_layer = self.transpose_for_scores(k)
        v_layer = self.transpose_for_scores(v)

        ca_scores = torch.matmul(q_layer.transpose(2, 3), k_layer) / math.sqrt(self.attention_head_size)
        # b, h, d/h, d/h

        ca_attn_probs = self.dropout(F.softmax(ca_scores, dim=-1))
        ca_attn_outputs = torch.matmul(ca_attn_probs, v_layer.transpose(2, 3)).permute(0, 3, 1, 2).contiguous()
        # b, l, h, d/h

        return ca_attn_outputs.view(x.size(0), -1, self.num_attention_heads * self.attention_head_size)

class ChannelAttentionBlock(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.channel_attention = ChannelAttention(config)

    def forward(self, x):
        attention_output = self.attention(x)
        channel_attention_output = self.channel_attention(x)
        return self.feed_forward(attention_output + channel_attention_output)

class CABertModel(BertPreTrainedModel):
    def __init__(self, config, pooling=True):
        super().__init__(config)
        self.config = config
        self.pooling = pooling

        self.embedding = BertEmbedding(config)
        self.transformer_blocks = nn.ModuleList(
            [ChannelAttentionBlock(config) for _ in range(config.num_hidden_layers)]
        )

        if pooling:
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
        pooled_output = self.tanh(self.dense(x[:, 0])) if self.pooling else None
        return x, pooled_output

class CABertNoNSP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = CABertModel(config, pooling=False)
        self.mlm = MaskedLanguageModel(config)

        self.post_init()

    def get_output_embeddings(self):
        return self.mlm.mlm_linear

    def set_output_embeddings(self, new_embeddings):
        self.mlm.mlm_linear = new_embeddings

    def forward(self, x):
        bert_output, _ = self.bert(x)

        mlm_output = self.mlm(bert_output)

        loss_func = CrossEntropyLoss()
        mlm_loss = loss_func(mlm_output.view(-1, self.config.vocab_size), x["labels"].view(-1))

        return mlm_loss