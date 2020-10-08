from NeuronalTransformerLayers.NeuronBank import NeuronBank
from NeuronalTransformerLayers.NeuronEncoder import NeuronEncoder
from NeuronalTransformerLayers.NeuronDecoder import NeuronDecoder

import torch
import torch.nn as nn

class NeuronalLayer(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronalLayer, self).__init__()
        self.encoder = NeuronEncoder(roberta_config, neuron_config)
        self.neuron_bank = NeuronBank(neuron_config)
        self.decoder = NeuronDecoder(roberta_config, neuron_config)
        self.layer_norm = nn.LayerNorm(neuron_config.expanded_size, eps=roberta_config.layer_norm_eps)

    def forward(self, hidden_states, neuron_states=None, neuron_bank_connectivity_sub=0, output_connectivity_sub=0):
        hidden_states_norm = self.layer_norm(hidden_states)

        encoded_inputs = self.encoder(hidden_states_norm)

        if neuron_states == None:
            input_states = encoded_inputs
        else:
            input_states = [torch.cat((inp, x), dim=1) for inp, x in zip(encoded_inputs, neuron_states)]

        out_neuron_states = self.neuron_bank(input_states, neuron_bank_connectivity_sub)

        out_states_addition = self.decoder(hidden_states_norm, out_neuron_states, output_connectivity_sub)

        out_states = hidden_states + out_states_addition

        return (out_states, out_neuron_states)