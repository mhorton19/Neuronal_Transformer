import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn

from NeuronalTransformerLayers.NeuronBank import NeuronBank
from NeuronalTransformerLayers.NeuronEncoder import NeuronEncoder
from NeuronalTransformerLayers.NeuronDecoder import NeuronDecoder

#ADD RESIDUAL CONNECTIONS (repeatedly feeding embeddings to each neuron layer)

class NeuronalTransformer(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronalTransformer, self).__init__()

        self.encoder = NeuronEncoder(roberta_config, neuron_config)
        self.neuron_bank = NeuronBank(neuron_config)
        self.decoder = NeuronDecoder(roberta_config, neuron_config)
        self.num_iterations = neuron_config.num_iterations

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
        encoded_inputs = self.encoder(hidden_states)

        x = self.neuron_bank(encoded_inputs, self_connection=False)
        for i in range(self.num_iterations-1):
            concatenated_hidden_states = [torch.cat((x, inp), dim=1) for x, inp in zip(x, encoded_inputs)]
            x = self.neuron_bank(concatenated_hidden_states, self_connection=True)

        x = self.decoder(hidden_states, x)

        return x

