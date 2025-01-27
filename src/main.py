import tiktoken as tk
import safetensors
import numpy as np
import torch
from torch.nn import functional as F


encd = tk.encoding_for_model('gpt2') # gets the encoding scheme for gpt2 wieghts depend on this encoding
f = open('data/tiny.txt', 'r')
tokens = encd.encode(f.read()) # encodes the text
# print(tokens)

x = torch.tensor(tokens[:64]) # input
y_target = torch.tensor(tokens[1:65]) # target
dim_model = 768
dim_k = 64

f = open('data/model.safetensors', 'rb')
fts = safetensors.deserialize(f.read())

# convert to torch tensor (map from string to tensor)
params = dict()

for ft in fts:
    name = ft[0]
    data = ft[1]['data']
    shape = ft[1]['shape']
    params[name] = torch.frombuffer(data, dtype=torch.float32).reshape(shape)

wte_out = F.embedding(x, params["wte.weight"])
wpe_out = F.embedding(torch.arange(len(x)),params["wpe.weight"])
embd_out = wte_out + wpe_out # tensor(-30.5272)
print(embd_out.sum())
exit(0)

for layer_i in range(12):
    ln_1_in = embd_out if layer_i == 0 else res_2_out

    ln_1_out = F.layer_norm(
        input=ln_1_in,
        normalized_shape=[embd_out.shape[-1]],
        weight=params[f"h.{layer_i}.ln_1.weight"],
        bias=params[f"h.{layer_i}.ln_1.bias"])

    # attention block
    # qkv
    # parameters from the model file are transposed 

    attn_c_attn_out = F.linear(
        input=ln_1_out, 
        weight=params[f"h.{layer_i}.attn.c_attn.weight"].transpose(0, 1), 
        bias=params[f"h.{layer_i}.attn.c_attn.bias"])

    q, k, v = attn_c_attn_out.split(dim_model, dim=1)
    attn_z_out = torch.zeros_like(ln_1_out)
    for head_i in range(1):
        a = q[:, head_i*dim_k:(head_i+1)*dim_k] @ k[:, head_i*dim_k:(head_i+1)*dim_k].transpose(0, 1)
        a /= torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))
        mask = torch.triu(torch.ones_like(a, dtype=bool), diagonal=1) # lower triangular mask
        a = torch.masked_fill(a, mask, -torch.inf) # fill the upper triangular with -inf
        s = F.softmax(a, dim=1)
        z = s @ v[:, head_i*dim_k:(head_i+1)*dim_k]
        attn_z_out[:, head_i*dim_k:(head_i+1)*dim_k] = z                                                                      

    attn_c_proj_out = F.linear(
        input=attn_z_out, 
        weight=params[f"h.{layer_i}.attn.c_proj.weight"].transpose(0, 1), 
        bias=params[f"h.{layer_i}.attn.c_proj.bias"])


    # Residual connection
    res_1_out = ln_1_in + attn_c_proj_out
   
    # Layer normalization
    ln_2_out = F.layer_norm(
        input=res_1_out,
        normalized_shape=[dim_model],
        weight=params[f"h.{layer_i}.ln_2.weight"],
        bias=params[f"h.{layer_i}.ln_2.bias"])

    # mlp block

    mlp_in = ln_2_out
    mlp_c_fc_out = F.linear(
        input=mlp_in,
        weight=params[f"h.{layer_i}.mlp.c_fc.weight"].transpose(0, 1),
        bias=params[f"h.{layer_i}.mlp.c_fc.bias"])

    # Gelu
    mlp_gelu_out = F.gelu(mlp_c_fc_out)

    mlp_c_proj_out = F.linear(
        input=mlp_gelu_out,
        weight=params[f"h.{layer_i}.mlp.c_proj.weight"].transpose(0, 1),
        bias=params[f"h.{layer_i}.mlp.c_proj.bias"])

    

    res_2_out = res_1_out + mlp_c_proj_out

ln_f_out = F.layer_norm(
    input=res_2_out,
    normalized_shape=[dim_model],
    weight=params["ln_f.weight"],
    bias=params["ln_f.bias"])

unembdd_out = F.linear(
    input=ln_f_out,
    weight=params["wte.weight"])

print(unembdd_out.shape)

token_idx = torch.argmax(unembdd_out[-1,:], dim=-1)
print(token_idx)
print(encd.decode(list(x)))
print(encd.decode([token_idx]))

#print cross entropy loss for the last token
ce_loss = F.cross_entropy(unembdd_out, y_target)
print(ce_loss)
