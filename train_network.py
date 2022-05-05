import os
import gym
import argparse
import numpy as np
from evolutionary_algorithms.classes.EA import EA
from evolutionary_algorithms.classes.Recombination import *
from evolutionary_algorithms.classes.Mutation import *
from evolutionary_algorithms.classes.Selection import *
from Evaluation import *
from Network import *

def main():
    parser = argparse.ArgumentParser()
    # evolutionary algorithms parameter
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
                        dest='eval_reps', type=int,
                        default=100,
                        help="Defines the number of evaluation repetitions to run after \
                                'training' our candidate individuals.")
    parser.add_argument('-env', action='store', type=str,
                        dest='env', default='CartPole-v1')
    parser.add_argument('-render_eval', action='store_true',
                        help='use this flag to render the evaluation process \
                                after training our individuals')
    parser.add_argument('-virtual_display', action='store_true',
                        help='needed for headless servers when using render')
    parser.add_argument('-v', action='store',
                        dest='verbose', type=int,
                        default=1,
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
    print(f"environment: {env.unwrapped.spec.id}")

    # define the individual size as the product of number of observations and
    # the number of possible actions
    n_observations = np.sum([dim for dim in env.observation_space.shape]) 
    n_actions = env.action_space.n

    # we create an istance here to define the individual size
    model = NN_regression(n_observations, 4, 4, n_actions).to("cpu")
    
    # define es individual size
    individual_size = model.total_params

    if args.verbose > 0:
        print(f"Individual size: {individual_size}, \
            n observations: {n_observations}, \
            n actions: {n_actions}")

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
    evaluation = RewardMaximizationNN(env, model, reps=10)

    # loop through experiment 
    best_results = []
    for _ in range(args.exp_reps):
        # define new ea istance
        ea = EA(minimize=minimize, budget=budget, patience=patience, 
        parents_size=parents_size, offspring_size=offspring_size,
        individual_size=individual_size, recombination=recombination,
        mutation=mutation, selection=selection, evaluation=evaluation,
        verbose=args.verbose)

        # run the ea and append results
        best_ind, best_eval = ea.run()
        best_results.append([best_ind, best_eval])

    # loop through final evalutation process for our best results
    eval_results = []
    for res in best_results:
        curr_eval = eval(res[0], env, model,
                        args.eval_reps, 
                        render=args.render_eval)
        eval_results.append([curr_eval])
    print("Evaluation results",eval_results)
    
    # initialize directory to save results
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save best individual in results directory
    best_ind_idx = np.argmax(eval_results)
    best_ind = best_results[best_ind_idx][0]
    np.save('results/'+args.exp_name+'.npy', best_ind)


def eval(individual, env, model, reps, render=False):
    """ Test evaluation function with repetitions
    """
    n_observations = np.sum([dim for dim in env.observation_space.shape]) 
    n_actions = env.action_space.n

    model.update_weights(individual)
    # loop through evaluation repetitions
    rews = []
    for _ in range(reps):
        done = False
        tot_rew = 0
        state = torch.tensor(env.reset(), requires_grad=False)
        if render:
            env.render()
        
        # loop through episode
        while not done:
            # TODO: Create sample action based on policy
            # sample action
            a = np.argmax(model(state)).numpy()
            # query environment
            state, rew, done, _ = env.step(a)
            state = torch.tensor(state, requires_grad=False)

            tot_rew += rew
            if render:
                env.render()
        rews.append(tot_rew)

    return np.mean(rews)



if __name__ == "__main__":
    main()