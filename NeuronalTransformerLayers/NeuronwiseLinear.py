import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def neuronwise_kaiming_uniform(tensor, fan, a=0, nonlinearity='leaky_relu'):
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class NeuronwiseLinear(nn.Module):
    def __init__(self, num_neurons: int, in_features: int, out_features: int, bias: bool = True, batch_first: bool = False) -> None:
        super(NeuronwiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons

        self.use_bias = bias
        self.batch_first = batch_first

        self.weight = nn.Parameter(torch.Tensor(num_neurons, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_neurons, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        neuronwise_kaiming_uniform(self.weight, self.out_features, a=math.sqrt(5))
        if self.use_bias:
            fan_in = self.out_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            input = input.permute((1,0,2))

        transformed = torch.matmul(input, self.weight)

        if self.use_bias:
            transformed += self.bias

        if self.batch_first:
            transformed = transformed.permute((1,0,2))

        return transformed

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )