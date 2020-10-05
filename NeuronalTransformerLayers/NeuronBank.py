import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn

from NeuronalTransformerLayers.NeuronwiseLinear import NeuronwiseLinear
import math

def scaled_softmax(attention_vals, scale_vals, dim=-1):
    attention_exp = torch.exp(attention_vals)
    attention_exp_scaled = attention_exp * scale_vals
    print('\nMEAN: ' + str(float(scale_vals.mean())) + '    STD: ' + str(float(scale_vals.std())))
    softmax_val = attention_exp_scaled / torch.sum(attention_exp_scaled, dim, keepdim=True)

    return softmax_val


class NeuronBank(nn.Module):
    def __init__(self, config):
        super(NeuronBank, self).__init__()

        self.num_neurons = config.num_neurons
        self.values_len = config.values_len
        self.query_len = config.query_len
        self.num_heads = config.num_heads

        self.internal_vec_size = self.values_len * self.num_heads

        self.use_connectivity = config.use_connectivity

        if self.use_connectivity:
            self.connectivity_scalars = torch.nn.Parameter(
                torch.Tensor(1, self.num_heads, self.num_neurons, self.num_neurons))

        self.query_bank = torch.nn.Parameter(torch.Tensor(self.num_heads * self.num_neurons, self.query_len))

        self.values_out = NeuronwiseLinear(self.num_neurons, self.values_len * self.num_heads, self.values_len * self.num_heads)
        self.keys_out = NeuronwiseLinear(self.num_neurons, self.values_len * self.num_heads, self.query_len)

        self.layer_norm = torch.nn.LayerNorm(self.values_len * self.num_heads, eps=config.layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_bank)
        if self.use_connectivity:
            nn.init.normal_(self.connectivity_scalars)

    def separate_attention_heads(self, hidden_state):
        hidden_state_shape = hidden_state.shape
        hidden_state_reshaped = hidden_state.view(hidden_state_shape[0], hidden_state_shape[1], self.values_len, self.num_heads)

        return hidden_state_reshaped

    def reshape_outputs(self, output_state):
        return output_state.permute(1, 0, 2).contiguous()

    #takes input in the format (batch size, num_inputs, values_len*num_heads) for values and (batch size, num_inputs, query_len) for keys
    def forward(self, hidden_states, self_connection=False):
        hidden_keys = hidden_states[0]
        hidden_values = hidden_states[1]

        # shape: (query_len, num_in, bs)
        keys_reshaped = hidden_keys.permute(2, 1, 0).contiguous()

        keys_reshaped_shape = keys_reshaped.shape
        # shape: (query_len, num_in*bs)
        keys_reshaped = keys_reshaped.view(keys_reshaped_shape[0], keys_reshaped_shape[1]*keys_reshaped_shape[2])

        # shape: (num_heads * num_out, num_in*bs)
        attention_scalars = torch.matmul(self.query_bank, keys_reshaped) / math.sqrt(self.query_len)

        attention_scalars_reshaped = attention_scalars.view(self.num_heads, self.num_neurons, keys_reshaped_shape[1], keys_reshaped_shape[2])
        # shape: (bs, num_heads, num_out, num_in)
        attention_scalars_reshaped = attention_scalars_reshaped.permute(3, 0, 1, 2).contiguous()

        if self_connection and self.use_connectivity:
            connectivity_matrix = torch.sigmoid(self.connectivity_scalars)
            attention_probs = scaled_softmax(attention_scalars_reshaped, connectivity_matrix, dim=-1)
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scalars_reshaped)


        values_reshaped = self.separate_attention_heads(hidden_values)
        # shape: (bs, num_heads, num_in, vec_size)
        values_reshaped = values_reshaped.permute(0, 3, 1, 2).contiguous()

        # shape: (bs, num_heads, num_out, vec_size)
        attention_outputs = torch.matmul(attention_probs, values_reshaped)


        attention_outputs_reshaped = attention_outputs.permute(0, 2, 3, 1).contiguous()
        output_values_reshaped_shape = attention_outputs_reshaped.shape
        # shape: (bs, num_out, vec_size*num_heads)
        attention_outputs_reshaped = attention_outputs_reshaped.view(*output_values_reshaped_shape[:2], output_values_reshaped_shape[2]*output_values_reshaped_shape[3])
        attention_outputs_reshaped = self.layer_norm(attention_outputs_reshaped)
        # shape: (num_out, bs, vec_size*num_heads)
        attention_outputs_reshaped = attention_outputs_reshaped.permute(1, 0, 2).contiguous()

        output_keys = self.reshape_outputs(self.keys_out(attention_outputs_reshaped))
        output_values = self.reshape_outputs(self.values_out(attention_outputs_reshaped))

        return (output_keys, output_values)
'''
import numpy as np
from NeuronalTransformerLayers.NeuronBankConfig import NeuronBankConfig

config = NeuronBankConfig()
neuron_bank = NeuronBank(config)

input_keys = torch.Tensor(np.random.normal(size=(8, 13, config.vec_size*config.num_heads)))
input_values = torch.Tensor(np.random.normal(size=(8, 13, config.vec_size*config.num_heads)))

neuron_bank((input_keys, input_values))'''