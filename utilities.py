import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def eval(env, model, reps, render=False):
    """ Test evaluation function with repetitions to average results.
    """
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

def save_plot(plot_name, optimal_val, data):
    """ Save plot of the performance of the algorithm
        for current evaluation function.
    """
    # create directory for plots and save plot
    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.clf() # clear past figures
    plt.plot(data.mean(axis=0), label=plot_name)
    plt.fill_between(np.arange(data.shape[1]),data.min(axis=0), 
                                data.max(axis=0),alpha=0.2)
    plt.axhline(y=optimal_val, xmin=0, xmax=1, color='r', linestyle='--')
    plt.xlabel("generation")
    plt.ylabel("evaluation")
    plt.title(plot_name)
    plt.savefig('plots/'+plot_name)