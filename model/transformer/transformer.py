import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), -1, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x) # b, l, d

        q_layer = self.transpose_for_scores(q)  # b, h, l, d/h
        k_layer = self.transpose_for_scores(k)
        v_layer = self.transpose_for_scores(v)

        attn_scores = torch.matmul(q_layer, k_layer.transpose(2, 3)) / math.sqrt(self.attention_head_size)
        # b, h, l, l

        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))
        attn_outputs = torch.matmul(attn_probs, v_layer).permute(0, 2, 1, 3).contiguous()
        # b, l, h, d/h

        return attn_outputs.view(x.size(0), -1, self.num_attention_heads * self.attention_head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        attn_output = self.attention(x)
        return self.layer_norm(self.dropout(self.dense(attn_output)) + x)

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.gelu = nn.GELU()

    def forward(self, x):
        intermediate_output = self.gelu(self.linear_1(x))
        return self.layer_norm(self.dropout(self.linear_2(intermediate_output)) + x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        attention_output = self.attention(x)
        return self.feed_forward(attention_output)

if __name__ == "__main__":
    pass