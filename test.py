import os
import gym
import torch
import argparse
from Network import *
import numpy as np

def main():
    model = NN_regression(  input_size=4,
                            input_neurons=4,
                            hidden_neurons=4,
                            output_size=2)

    print("Print network details test")
    model_tot_params = 0
    for name, params in model.named_parameters():
        model_tot_params += params.numel()
        print(f"{name} - size: {params.numel()}, shape: {params.shape}")
        print(params)
    print(f"total model parameters: {model_tot_params}")

    print("\n Model params substitution test")
    new_weights_dict = {}
    new_weights = np.random.uniform(size=model_tot_params)
    for name, params in model.named_parameters():

        layer_weights = torch.tensor(new_weights[:params.numel()].reshape(params.shape))
        new_weights_dict[name] = layer_weights
    model.load_state_dict(new_weights_dict)
    for name, params in model.named_parameters():
        print(params)
if __name__ == "__main__":
    main()