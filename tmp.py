from model.bert.Bert import BertNoNSP as our_model
from model.bert.Bert_HF import BertForMaskedLM as hf_model
from transformers.models.bert.configuration_bert import BertConfig

BERT_cfg = {
    # prajjwal1/bert-     [n_embd, n_layer]
    "prajjwal1/bert-tiny": [128, 2],
    "prajjwal1/bert-mini": [256, 4],
    "prajjwal1/bert-small": [512, 4],
    "prajjwal1/bert-medium": [512, 8],
}



cfg = BertConfig(hidden_size=BERT_cfg["prajjwal1/bert-tiny"][0], num_hidden_layers=BERT_cfg["prajjwal1/bert-tiny"][1],
                      num_attention_heads=BERT_cfg["prajjwal1/bert-tiny"][1], attention_probs_dropout_prob=0.1)

def info(model):
    param_size = 0
    nparam = 0
    sparam = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        nparam += param.nelement()
        sparam += param.element_size()
    print("param size: {:.3f}MB".format(param_size / 1024 ** 2))
    print(f"num param , size param { nparam , sparam}")
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    print("buffer size: {:.3f}MB".format(buffer_size / 1024 ** 2))
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))

print(cfg)

our = our_model(cfg)
hf = hf_model(cfg)

our.train()
hf.train()

# print(hf.state_dict().keys())

our_key = list(our.state_dict().keys())
hf_key = list(hf.state_dict().keys())
print(our_key, hf_key)

for i in range(len(hf_key)):
    print(our_key[i], "  //  ", hf_key[i])


# for i in range(44):
#     try:
#         print(our_key[i], hf_key[i])
#     except:
#         print(hf_key[i])

# print('-'*10)
# for name, module in our.named_parameters():
#     if 'mlm_linear' in name:
#         print(name, module.shape)

# print('-'*10)
# p = 0
# for name, module in hf.named_parameters():
#     p += module.numel()
#     print(name)

print(sum(p.numel() for p in our.parameters() if p.requires_grad))
print(sum(p.numel() for p in hf.parameters() if p.requires_grad))

print(sum(p.numel() for p in our.bert.parameters() if p.requires_grad))
print(sum(p.numel() for p in hf.bert.parameters() if p.requires_grad))

print(sum(p.numel() for p in our.mlm.parameters() if p.requires_grad))
print(sum(p.numel() for p in hf.cls.parameters() if p.requires_grad))

info(our)
info(hf)