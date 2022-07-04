import torch
from torch import nn

class NN_base(nn.Module):
    def __init__(self) -> None:
        super(NN_base, self).__init__()
        self.layers = nn.Sequential()
        self.total_params = 0

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)

    def update_weights(self,new_weights):
        # create new dictionary of weights
        new_weights_dict = {}
        for name, params in self.named_parameters():
            layer_weights = torch.tensor(new_weights[:params.numel()].reshape(params.shape),
                                    requires_grad=False)
            new_weights_dict[name] = layer_weights
            # discard weights used
            new_weights =  new_weights[params.numel():]
        # substitute new weights in the model
        self.load_state_dict(new_weights_dict)


class NN_regression_0(NN_base):
    """ Network with 0 hidden layers.
    """
    def __init__(self, input_size, input_neurons, hidden_neurons, output_size):
        super(NN_regression_0, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False)
        )
        self.total_params = sum(p.numel() for p in self.parameters())


class NN_regression_1(NN_base):
    """ Network with 1 hidden layer and 1 ReLU activation.
    """
    def __init__(self, input_size, input_neurons, hidden_neurons, output_size):
        super(NN_regression_1, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, input_neurons),
            nn.ReLU(),
            nn.Linear(input_neurons, output_size)
        )
        self.total_params = sum(p.numel() for p in self.parameters())


class NN_regression_2(NN_base):
    """ Network with 2 hidden layers
    """
    def __init__(self, input_size, input_neurons, hidden_neurons, output_size):
        super(NN_regression_2, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, input_neurons),
            nn.ReLU(),
            nn.Linear(input_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_size)
        )
        self.total_params = sum(p.numel() for p in self.parameters())

class NN_regression_3(NN_base):
    """ Network with 1 hidden layer and 1 ReLU activation.
    """
    def __init__(self, input_size, input_neurons, hidden_neurons, output_size):
        super(NN_regression_3, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, input_neurons),
            nn.Tanh(),
            nn.Linear(input_neurons, output_size),
        )
        self.total_params = sum(p.numel() for p in self.parameters())