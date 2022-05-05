import torch
from torch import nn


class NN_regression(nn.Module):
    """ Convolution model that returns one value as output.
    """
    def __init__(self, input_size, input_neurons, hidden_neurons, output_size):
        super(NN_regression, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, input_neurons),
            nn.ReLU(),
            nn.Linear(input_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_size)
        )
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.layers(x)


    def update_weights(self,new_weights):
        # create new dictionary of weights
        new_weights_dict = {}
        for name, params in self.named_parameters():
            layer_weights = torch.tensor(new_weights[:params.numel()].reshape(params.shape))
            new_weights_dict[name] = layer_weights
        # substitute new weights in the model
        self.load_state_dict(new_weights_dict)

