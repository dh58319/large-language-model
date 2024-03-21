import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from

class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden = config.hidden
        self.n_layers = config.n_layers
        self.attn_heads = config.attention_heads

        self.feed_forward_hidden = config.hidden * 4
        self.embedding = self.BERTEmbedding(vocab_size=config.vocab_size, embed_size=self.hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

class NextSentencePrediction(nn.Module):
    pass

class MaskedLanguageModel(nn.Module):
    pass

class BERTforLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BERTModel(config)
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, config.vocab_size)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        ## inputs (from Wikitext) : {input_ids, token_type_ids, attention_mask, labels}
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)



