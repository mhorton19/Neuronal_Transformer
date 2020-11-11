from NeuronalTransformerLayers.NeuronBank import NeuronBank
from NeuronalTransformerLayers.NeuronEncoder import NeuronEncoder
from NeuronalTransformerLayers.NeuronDecoder import NeuronDecoder
from NeuronalTransformerLayers.GenNeuronStates import GenNeuronStates
from NeuronalTransformerLayers.GenNeuronOutputs import GenNeuronOutputs

import torch
import torch.nn as nn

class NeuronalLayer(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronalLayer, self).__init__()
        self.num_groups = neuron_config.num_groups

        self.encoder = NeuronEncoder(roberta_config, neuron_config)
        self.gen_neuron_states = GenNeuronStates(neuron_config)
        self.gen_neuron_outputs = GenNeuronOutputs(neuron_config, num_groups=self.num_groups)

        self.decoder = NeuronDecoder(roberta_config, neuron_config)
        self.layer_norm = nn.LayerNorm(neuron_config.expanded_size, eps=roberta_config.layer_norm_eps)

        self.num_neurons_per_group = neuron_config.num_neurons_per_group
        self.values_len = neuron_config.values_len
        self.num_heads = neuron_config.num_heads



        self.num_neurons = self.num_groups * self.num_neurons_per_group

    def forward(self, hidden_states, prev_neuron_outputs=None, neuron_bank_connectivity_sub=0, output_connectivity_sub=0):
        hidden_states_norm = self.layer_norm(hidden_states)

        input_keys, input_values = self.encoder(hidden_states_norm)

        if prev_neuron_outputs == None:
            prev_neuron_outputs = self.gen_neuron_outputs(torch.normal(0, 1, size=(hidden_states.shape[0], self.num_groups, self.num_neurons_per_group, self.values_len * self.num_heads), device=hidden_states.device))

        neuron_keys, neuron_queries, neuron_values = prev_neuron_outputs
        keys = torch.cat((input_keys, neuron_keys), dim=2)
        values = torch.cat((input_values, neuron_values), dim=2)


        neuron_states = self.gen_neuron_states(neuron_queries, keys, values, neuron_bank_connectivity_sub)
        neuron_outputs = self.gen_neuron_outputs(neuron_states)

        out_states_addition = self.decoder(hidden_states_norm, neuron_outputs, output_connectivity_sub)

        out_states = hidden_states + out_states_addition

        return (out_states, neuron_outputs)