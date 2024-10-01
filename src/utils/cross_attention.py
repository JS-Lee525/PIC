import torch
import torch.nn as nn
from diffusers.models.attention import CrossAttention

class MyConv2D(nn.Module):
    def __init__(self, c):
        super(). __init__()
        self.c = c
        
    def forward(self, input):
        self.h = self.c(input)
        
        return self.h

class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):

    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        if 'attn2' in name:
            params.requires_grad = False
        else:
            params.requires_grad = False
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.set_processor(MyCrossAttnProcessor())
        if name == 'mid_block.resnets.1.conv2':
            c = unet.mid_block.resnets[1].conv2
            unet.mid_block.resnets[1].conv2 = MyConv2D(c)
    return unet