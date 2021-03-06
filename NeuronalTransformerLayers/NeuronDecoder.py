import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import math

class NeuronDecoder(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronDecoder, self).__init__()

        self.vec_size = neuron_config.vec_size
        self.num_heads = neuron_config.num_heads
        self.out_len = self.vec_size * self.num_heads
        self.num_duplicates = neuron_config.num_duplicates
        self.query_linear = nn.Linear(self.out_len * self.num_duplicates, roberta_config.hidden_size)
        self.reembed_linear = nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size)

        self.layer_norm = torch.nn.LayerNorm(roberta_config.hidden_size, eps=neuron_config.layer_norm_eps)

    def separate_attention_heads(self, hidden_state):
        hidden_state_shape = hidden_state.shape
        hidden_state_reshaped = hidden_state.view(hidden_state_shape[0], hidden_state_shape[1], self.vec_size, self.num_heads)

        return hidden_state_reshaped

    def forward(self, embeddings, hidden_states):
        embeddings_shape = embeddings.shape
        queries = self.query_linear(embeddings)
        queries = queries.view(embeddings_shape[0], embeddings_shape[1] * self.num_duplicates, self.out_len)
        queries_shape = queries.shape
        queries = queries.view(*queries_shape[:2], self.vec_size, self.num_heads)
        queries = queries.permute(0, 3, 1, 2).contiguous()

        keys, values = hidden_states

        # shape: bs, num_heads, vec_size, num_in
        keys_reshaped = self.separate_attention_heads(keys).permute(0, 3, 2, 1).contiguous()

        # shape: bs, num_heads, num_in, vec_size
        values_reshaped = self.separate_attention_heads(values).permute(0, 3, 1, 2).contiguous()

        attention_scores = torch.matmul(queries, keys_reshaped)
        attention_scores = attention_scores / math.sqrt(self.vec_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        values_out = torch.matmul(attention_probs, values_reshaped)

        values_out_shape = values_out.shape
        values_out_reshaped = values_out.view(*values_out_shape[:2], -1, values_out_shape[3] * self.num_duplicates)
        values_out_reshaped = values_out_reshaped.permute(0, 2, 1, 3).contiguous()
        values_out_reshaped = values_out_reshaped.view(*values_out_reshaped.shape[:2], -1)

        reembedded = self.reembed_linear(self.layer_norm(values_out_reshaped))
        return reembedded