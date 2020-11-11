import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import math

class NeuronDecoder(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronDecoder, self).__init__()

        self.values_len = neuron_config.values_len
        self.num_heads = neuron_config.num_heads
        self.num_duplicates = neuron_config.num_duplicates
        self.query_len = neuron_config.query_len
        self.num_groups = neuron_config.num_groups

        self.query_linear = nn.Linear(neuron_config.expanded_size, self.num_groups * self.query_len * self.num_heads * self.num_duplicates)
        self.reembed_linear = nn.Linear(self.num_groups * self.values_len * self.num_heads * self.num_duplicates, neuron_config.expanded_size)

        self.layer_norm = torch.nn.LayerNorm(self.num_groups * self.values_len * self.num_heads * self.num_duplicates, eps=neuron_config.layer_norm_eps)

    def separate_attention_heads(self, hidden_state, vec_len):
        hidden_state_shape = hidden_state.shape
        hidden_state_reshaped = hidden_state.view(hidden_state_shape[0], hidden_state_shape[1], hidden_state_shape[2], vec_len, self.num_heads)

        return hidden_state_reshaped

    # takes input in the format (batch size, num_groups, num_inputs, values_len*num_heads) for values and (batch size, num_groups, num_inputs, query_len*num_heads) for keys
    def forward(self, embeddings, neuron_outputs, output_connectivity_sub):
        embeddings_shape = embeddings.shape
        queries = self.query_linear(embeddings)
        queries = queries.view(embeddings_shape[0], self.num_groups, embeddings_shape[1] * self.num_duplicates, self.query_len * self.num_heads)
        queries_shape = queries.shape
        queries = queries.view(*queries_shape[:3], self.query_len, self.num_heads)
        queries = queries.permute(0, 1, 4, 2, 3).contiguous()

        _, keys, values = neuron_outputs

        # shape: bs, num_groups, num_heads, query_len, num_in
        keys_reshaped = self.separate_attention_heads(keys, self.query_len).permute(0, 1, 4, 3, 2).contiguous()

        # shape: bs, num_groups, num_heads, num_in, values_len
        values_reshaped = self.separate_attention_heads(values, self.values_len).permute(0, 1, 4, 2, 3).contiguous()

        attention_scores = torch.matmul(queries, keys_reshaped)
        attention_scores = attention_scores / math.sqrt(self.query_len)

        attention_probs = nn.Softmax(dim=-1)(attention_scores - output_connectivity_sub)

        values_out = torch.matmul(attention_probs, values_reshaped)

        values_out_shape = values_out.shape
        values_out_reshaped = values_out.view(*values_out_shape[:3], self.num_duplicates, embeddings_shape[1], values_out_shape[-1])
        values_out_reshaped = values_out_reshaped.permute(0, 4, 1, 2, 3, 5).contiguous()
        values_out_reshaped = values_out_reshaped.view(*values_out_reshaped.shape[:2], -1)

        reembedded = self.reembed_linear(self.layer_norm(values_out_reshaped))
        return reembedded