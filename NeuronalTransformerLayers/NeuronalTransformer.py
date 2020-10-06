import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn

from NeuronalTransformerLayers.NeuronBank import NeuronBank
from NeuronalTransformerLayers.NeuronEncoder import NeuronEncoder
from NeuronalTransformerLayers.NeuronDecoder import NeuronDecoder
from NeuronalTransformerLayers.NeuronalLayer import NeuronalLayer

import torch
import torch.nn as nn

#ADD RESIDUAL CONNECTIONS (repeatedly feeding embeddings to each neuron layer)

class NeuronalTransformer(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronalTransformer, self).__init__()

        self.neuronal_layer = NeuronalLayer(roberta_config, neuron_config)
        self.num_iterations = neuron_config.num_iterations
        self.num_duplicates = neuron_config.num_duplicates

        self.embedding_transformation = nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size)

        self.layer_norm = nn.LayerNorm(roberta_config.hidden_size, eps=roberta_config.layer_norm_eps)

        self.use_connectivity = neuron_config.use_connectivity

        if self.use_connectivity:
            self.neural_connectivity_scalars = torch.nn.Parameter(
                torch.Tensor(1, neuron_config.num_heads, neuron_config.num_neurons, neuron_config.num_neurons))

            self.input_connectivity_scalars = torch.nn.Parameter(
                torch.Tensor(1, neuron_config.num_heads, neuron_config.num_neurons, 1))

            self.output_connectivity_scalars = torch.nn.Parameter(
                torch.Tensor(1, neuron_config.num_heads, 1, neuron_config.num_neurons))

        self.connectivity_coefficient = neuron_config.connectivity_coefficient
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_connectivity:
            nn.init.zeros_(self.neural_connectivity_scalars)
            nn.init.zeros_(self.input_connectivity_scalars)
            nn.init.zeros_(self.output_connectivity_scalars)

    def calc_connectivity_sub(self, scalars):
        return torch.exp(scalars * self.connectivity_coefficient)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        hidden_states_transformed = self.embedding_transformation(hidden_states)

        input_connectivity_sub = self.calc_connectivity_sub(self.input_connectivity_scalars)
        neural_connectivity_sub = self.calc_connectivity_sub(self.neural_connectivity_scalars)
        output_connectivity_sub = self.calc_connectivity_sub(self.output_connectivity_scalars)

        repeated_input_connectivity = input_connectivity_sub.expand(-1, -1, -1, hidden_states.shape[1] * self.num_duplicates)
        neuron_bank_connectivity_sub = torch.cat((repeated_input_connectivity, neural_connectivity_sub), axis=3)

        hidden_states_transformed, neuronal_states = self.neuronal_layer(hidden_states_transformed,
                                                                             neuron_bank_connectivity_sub=input_connectivity_sub,
                                                                             output_connectivity_sub=output_connectivity_sub)
        for i in range(self.num_iterations-1):
            hidden_states_transformed, neuronal_states = self.neuronal_layer(hidden_states_transformed,
                                                                             neuron_states=neuronal_states,
                                                                             neuron_bank_connectivity_sub=neuron_bank_connectivity_sub,
                                                                             output_connectivity_sub=output_connectivity_sub)



        return self.layer_norm(hidden_states_transformed)

