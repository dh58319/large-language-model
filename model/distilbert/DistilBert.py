import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddins = nn.Embedding(config.max_)

class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding = Embedding(config)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layer)]
        )

        self.post_init()


class DistilBertNoNSP(DistilBertPreTrainedModel):
    _tied_weights_keys = ["vocab_projector.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.distilbert = DistilBertModel(config)
