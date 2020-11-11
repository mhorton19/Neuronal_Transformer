import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn

from NeuronalTransformerLayers.NeuronwiseLinear import NeuronwiseLinear

class GenNeuronOutputs(nn.Module):
    def __init__(self, config, num_groups=1):
        super(GenNeuronOutputs, self).__init__()

        self.num_neurons_per_group = config.num_neurons_per_group
        self.values_len = config.values_len
        self.query_len = config.query_len
        self.num_heads = config.num_heads

        self.num_groups = num_groups

        self.num_neurons = self.num_groups * self.num_neurons_per_group

        self.values_out = NeuronwiseLinear(self.num_neurons, self.values_len * self.num_heads,
                                           self.values_len * self.num_heads, batch_first=True)

        self.keys_out = NeuronwiseLinear(self.num_neurons, self.values_len * self.num_heads,
                                         self.query_len * self.num_heads, batch_first=True)

        self.queries_out = NeuronwiseLinear(self.num_neurons, self.values_len * self.num_heads,
                                            self.query_len * self.num_heads, batch_first=True)

        self.layer_norm = torch.nn.LayerNorm(self.values_len * self.num_heads, eps=config.layer_norm_eps)

    def reshape_output(self, output):
        output_shape = output.shape
        return output.view(output_shape[0], self.num_groups, self.num_neurons_per_group, *output_shape[2:])

    def forward(self, neuron_states):
        normed_states = self.layer_norm(neuron_states.view(neuron_states.shape[0], self.num_neurons, *neuron_states.shape[3:]))

        queries = self.reshape_output(self.queries_out(normed_states))
        keys = self.reshape_output(self.keys_out(normed_states))
        values = self.reshape_output(self.values_out(normed_states))

        return (queries, keys, values)