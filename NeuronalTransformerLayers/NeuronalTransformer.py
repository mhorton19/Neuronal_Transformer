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

        self.embedding_transformation = nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size)

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

        hidden_states_transformed, neuronal_states = self.neuronal_layer(hidden_states_transformed)
        for i in range(self.num_iterations-1):
            hidden_states_transformed, neuronal_states = self.neuronal_layer(hidden_states_transformed, neuronal_states)

        return hidden_states_transformed

