import os
import gym
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    # gym parameters
    parser.add_argument('-env', action='store', type=str, default=None)

    # extra utility parameters
    parser.add_argument('-virtual_display', action='store_true')
    args = parser.parse_args()

    if args.virtual_display:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # create gym environment
    env = gym.make(parser.env) # 'CartPole-v1'
    if env is None:
        exit("Please select an environment")
    print(f"environment: {env.unwrapped.spec.id}")

    n_actions = env.action_space.n
    print(f"number of actions: {n_actions}")




if __name__ == "__main__":
    main()