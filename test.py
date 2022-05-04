import os
import gym
import torch
import argparse

def main():
    
    

    # create gym environment
    env = gym.make('CartPole-v1') # 'CartPole-v1'
    envs = []
    for i in range(1000):
        envs.append(gym.make('CartPole-v1'))
        envs[i].reset()
    env_2 = gym.make('CartPole-v1')
    if env is None:
        exit("Please select an environment")
    print(f"environment: {env.unwrapped.spec.id}")

    n_actions = env.action_space.n
    print(f"number of actions: {n_actions}")




if __name__ == "__main__":
    main()