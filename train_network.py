import os
import gym
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from evolutionary_algorithms.classes.EA import EA
from evolutionary_algorithms.classes.Recombination import *
from evolutionary_algorithms.classes.Mutation import *
from evolutionary_algorithms.classes.Selection import *
from classes.Evaluation import *
from classes.Network import *
from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_name', action='store',
                        dest='exp_name', type=str,
                        default='test_experiment',
                        help="Defines the name of the experiment.")
    parser.add_argument('-b', action='store',
                        dest='budget', type=int,
                        default=10000,
                        help="Defines the total amount of evaluations.")
    parser.add_argument('-min', action='store_true', 
                        dest='minimize',
                        help="Use this flag if the problem is minimization.")
    parser.add_argument('-r', action='store',
                        dest='recombination', type=str,
                        default=None,
                        help="Defines the recombination strategy.")
    parser.add_argument('-m', action='store',
                        dest='mutation', type=str,
                        default='IndividualSigma',
                        help="Defines the mutation strategy.")
    parser.add_argument('-s', action='store',
                        dest='selection', type=str,
                        default='PlusSelection',
                        help="Defines the selection strategy.")
    parser.add_argument('-ps', action='store',
                        dest='parents_size', type=int,
                        default=2,
                        help="Defines the number of parents per generation.")
    parser.add_argument('-os', action='store',
                        dest='offspring_size', type=int,
                        default=4,
                        help="Defines the number of offspring per generation.")
    parser.add_argument('-mul', action='store',
                        dest='one_fifth_mul', type=float,
                        default=0.9,
                        help="Defines the multiplier for the one fifth success rule.")
    parser.add_argument('-pat', action='store',
                        dest='patience', type=int,
                        default=None,
                        help="Defines the wait time before resetting sigmas.")          
    parser.add_argument('-exp_reps', action='store',
                        dest='exp_reps', type=int,
                        default=5,
                        help="Defines the number of experiments to average results.")
    parser.add_argument('-train_reps', action='store',
                        dest='train_reps', type=int,
                        default=10,
                        help="Defines the number of evaluation repetitions to use during training.")
    parser.add_argument('-eval_reps', action='store',
                        type=int, default=100,
                        help="Defines the number of evaluation repetitions to run after \
                                'training' our candidate individuals.")
    parser.add_argument('-model', action='store',
                        dest='model', type=int,
                        default=0,
                        help="Defines the model architecture to use.")
    parser.add_argument('-env', action='store', type=str,
                        dest='env', default='CartPole-v1')
    parser.add_argument('-render_train', action='store_true',
                        help='use this flag to render the training process \
                                of the individuals')
    parser.add_argument('-render_eval', action='store_true',
                        help='use this flag to render the evaluation process \
                                after training our individuals')
    parser.add_argument('-virtual_display', action='store_true',
                        help='needed for headless servers when using render')
    parser.add_argument('-plot_name', action='store', type=str,
                        default=None)
    parser.add_argument('-env_threshold', action='store',
                        type=float, default=500,
                        help="Optimum value to set as horizontal line in plot")
    parser.add_argument('-v', action='store',
                        dest='verbose', type=int, default=0,
                        help="Defines the intensity of debug prints.")
    args = parser.parse_args()
    if args.verbose > 0:
        print(args)

    # used to train on headless servers
    if args.virtual_display:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # create gym environment
    env = gym.make(args.env)
    if env is None:
        exit("Please select an environment")
    print()

    # environment specific parameters
    n_observations = np.sum([dim for dim in env.observation_space.shape])
    if env.action_space.__class__.__name__ == "Discrete":
        n_actions = env.action_space.n
    elif env.action_space.__class__.__name__ == "Box":
        n_actions = sum(env.action_space._shape)
    else:
        exit(f"{env.action_space.__class__.__name__} action space not yet implemented")

    # create an instance of the model
    if args.model == 0:
        model = NN_regression_0(n_observations, 4, 4, n_actions).to("cpu")
    elif args.model == 1:
        model = NN_regression_1(n_observations, 4, 4, n_actions).to("cpu")
    elif args.model == 2:
        model = NN_regression_2(n_observations, 4, 4, n_actions).to("cpu")

    # define es individual size
    individual_size = model.total_params

    print(f"Model architecture: {args.model}\nEnvironment: {env.unwrapped.spec.id}\nNumber of observations: {n_observations}\nNumber of actions: {n_actions}\nIndividual size: {individual_size}")
    print()

    # EA specific parameters
    minimize = args.minimize
    budget = args.budget
    patience = args.patience
    parents_size = args.parents_size
    offspring_size = args.offspring_size
    # Recombination specific controls
    if args.recombination != None:
        recombination = globals()[args.recombination]()
    elif args.recombination == "GlobalIntermediary" and args.offspring_size > 1:
        print("GlobalIntermediary recombination cannot be used with more than one offspring.")
        print("Please use a valid configuration")
        exit()
    else: recombination = None
    # Mutation specific controls
    if args.mutation == "IndividualOneFifth":
        mutation = globals()[args.mutation](args.one_fifth_mul)
    else:
        mutation = globals()[args.mutation]()
    selection=globals()[args.selection]()
    evaluation = RewardMaximizationNN(env, model, args.train_reps, 
                                        args.render_train)

    # loop through experiment 
    best_results = []
    data_for_plots = []
    for i in range(args.exp_reps):
        # define new ea istance
        ea = EA(minimize=minimize, budget=budget, patience=patience, 
        parents_size=parents_size, offspring_size=offspring_size,
        individual_size=individual_size, recombination=recombination,
        mutation=mutation, selection=selection, evaluation=evaluation,
        verbose=args.verbose)

        # run the ea
        start_time = time.time()
        best_ind, best_eval, all_best_evals = ea.run()
        end_time = time.time()

        # keep track of results
        best_results.append([best_ind, best_eval])
        data_for_plots.append(all_best_evals)
        print(f"Rep: {i+1} | best average of {args.exp_reps} evaluation reps: {best_eval} | time: {np.round(end_time-start_time, 2)}")

    # save plot if name has been defined
    if args.plot_name != None:
        save_plot(args.plot_name, args.env_threshold, np.array(data_for_plots))

    # loop through final evalutation process for our best results
    eval_results = []
    for res in best_results:
        model.update_weights(res[0])
        curr_eval = eval(env, model,
                        args.eval_reps, 
                        render=args.render_eval)
        eval_results.append(curr_eval)
    print("Evaluation results",np.round(eval_results, 2))
    
    # if saving is enabled
    if args.exp_name != None:
    # initialize directory and save results
        if not os.path.exists('model_weights'):
            os.makedirs('model_weights')
        # get the best individual
        best_ind_idx = np.argmax(eval_results)
        best_ind = best_results[best_ind_idx][0]
        # update model weights to match best inidividual
        model.update_weights(best_ind)
        # Save best individual as model weights
        torch.save(model.state_dict(), 'model_weights/'+args.exp_name)
        # Also save it as a numpy array
        np.save('model_weights/'+args.exp_name+'.npy', best_ind)

if __name__ == "__main__":
    main()