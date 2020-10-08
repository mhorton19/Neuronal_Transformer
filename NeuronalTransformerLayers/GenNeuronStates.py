import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn

import math

class GenNeuronStates(nn.Module):
    def __init__(self, config):
        super(GenNeuronStates, self).__init__()

        self.num_neurons = config.num_neurons
        self.values_len = config.values_len
        self.query_len = config.query_len
        self.num_heads = config.num_heads

    def separate_attention_heads(self, hidden_state, vec_len):
        hidden_state_shape = hidden_state.shape
        hidden_state_reshaped = hidden_state.view(hidden_state_shape[0], hidden_state_shape[1], vec_len, self.num_heads)

        return hidden_state_reshaped

    #queries: (bs, num_neurons, query_len*num_heads) keys: (bs, num_in, query_len*num_heads) values: (bs, num_in, values_len*num_heads)
    def forward(self, queries, keys, values, neuron_bank_connectivity_sub):
        queries_reshaped = self.separate_attention_heads(queries, self.query_len)
        queries_reshaped = queries_reshaped.permute(0, 3, 1, 2).contiguous()

        keys_reshaped = self.separate_attention_heads(keys, self.query_len)
        keys_reshaped = keys_reshaped.permute(0, 3, 2, 1).contiguous()

        # (bs, num_heads, num_neurons, num_in)
        attention_scalars = torch.matmul(queries_reshaped, keys_reshaped) / math.sqrt(self.query_len)

        attention_probs = nn.Softmax(dim=-1)(attention_scalars - neuron_bank_connectivity_sub)

        values_reshaped = self.separate_attention_heads(values, self.values_len)
        values_reshaped = values_reshaped.permute(0, 3, 1, 2).contiguous()

        # shape: (bs, num_heads, num_neurons, vec_size)
        attention_outputs = torch.matmul(attention_probs, values_reshaped)

        attention_outputs_reshaped = attention_outputs.permute(0, 2, 3, 1).contiguous()

        return attention_outputs.view(*attention_outputs_reshaped.shape[:2], -1)