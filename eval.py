import numpy as np
import argparse
import gym
import os
from classes.Network import *
from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', action='store',
                        dest='model', type=int,
                        default=0,
                        help="Defines the model architecture to use.")
    parser.add_argument('-model_weights', action='store',
                        type=str, default='test_experiment',
                        help="Defines the name of the experiment.")
    parser.add_argument('-env', action='store', type=str,
                        dest='env', default='CartPole-v1')
    parser.add_argument('-eval_reps', action='store',
                        type=int, default=100,
                        help="Number of times to evaluate our individual.")
    parser.add_argument('-render_eval', action='store_true',
                        help='use this flag to render the evaluation process \
                                after training our individuals')
    parser.add_argument('-virtual_display', action='store_true',
                        help='needed for headless servers when using render')
    args = parser.parse_args()
    print(args)


    # used to train on headless servers
    if args.virtual_display:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # create gym environment
    env = gym.make(args.env)
    if env is None:
        exit("Please select an environment")


    # environment specific parameters
    n_observations = np.sum([dim for dim in env.observation_space.shape]) 
    n_actions = env.action_space.n

    # create an instance of the model
    if args.model == 0:
        model = NN_regression_0(n_observations, 4, 4, n_actions).to("cpu")
    elif args.model == 1:
        model = NN_regression_1(n_observations, 4, 4, n_actions).to("cpu")
    elif args.model == 2:
        model = NN_regression_2(n_observations, 4, 4, n_actions).to("cpu")\

    # load weights
    model.load_state_dict(torch.load(args.model_weights))

    # evaluate
    mean_eval = eval( env, model,
                    args.eval_reps, 
                    render=args.render_eval)

    print(f"Evaluation mean: {np.round(mean_eval, 2)} for {args.eval_reps} repetitions.")

if __name__ == "__main__":
    main()