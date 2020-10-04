import sys
sys.path.insert(0, '../')

import torch.nn as nn

class NeuronEncoder(nn.Module):
    def __init__(self, roberta_config, neuron_config):
        super(NeuronEncoder, self).__init__()

        self.out_len = neuron_config.vec_size * neuron_config.num_heads
        self.num_duplicates = neuron_config.num_duplicates
        self.key_linear = nn.Linear(self.out_len * self.num_duplicates, roberta_config.hidden_size)
        self.value_linear = nn.Linear(self.out_len * self.num_duplicates, roberta_config.hidden_size)


    def forward(self, embeddings):
        embeddings_shape = embeddings.shape

        output_keys = self.key_linear(embeddings)
        output_keys = output_keys.view(embeddings_shape[0], embeddings_shape[1] * self.num_duplicates, self.out_len)

        output_vals = self.value_linear(embeddings)
        output_vals = output_vals.view(embeddings_shape[0], embeddings_shape[1] * self.num_duplicates, self.out_len)

        return (output_keys, output_vals)
