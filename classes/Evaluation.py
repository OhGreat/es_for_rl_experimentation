import numpy as np
import torch

from evolutionary_algorithms.classes.Population import Population


class RewardMaximizationNN():
    def __init__(self, env, model, reps=3, render=False):
        self.env = env
        self.model = model
        self.n_observations = np.sum([dim for dim in env.observation_space.shape]) 
        self.n_actions = env.action_space.n
        self.reps = reps
        self.render = render

    def __call__(self, population: Population):
        fitnesses = []
        for individual in population.individuals:
            ind_rews = []
            for i in range(self.reps):
                self.model.update_weights(individual)
                state = torch.tensor(self.env.reset(), requires_grad=False)
                if self.render:
                    self.env.render()
                rep_rews = 0
                done = False
                while not done:
                    # sample action
                    a = np.argmax(self.model(state)).numpy()
                    # query environment
                    state, rew, done, _ = self.env.step(a)
                    state = torch.tensor(state, requires_grad=False)
                    if self.render:
                        self.env.render()
                    rep_rews += rew
                ind_rews.append(rep_rews)
            fitnesses.append(np.mean(ind_rews))
        population.fitnesses = np.array(fitnesses)


class RewardMaximization():
    def __init__(self, env, reps=3):
        self.env = env
        self.n_observations = np.sum([dim for dim in env.observation_space.shape]) 
        self.n_actions = env.action_space.n
        self.reps = reps

    def __call__(self, population):
        fitnesses = []
        for individual in population.individuals:
            ind_rews = []
            for i in range(self.reps):
                state = self.env.reset()
                rep_rews = 0
                done = False
                while not done:
                    # sample action
                    a = np.argmax(np.dot(individual.reshape(self.n_actions, 
                                                        self.n_observations), 
                                                        state))                    
                    # query environment
                    state, rew, done, _ = self.env.step(a)
                    rep_rews += rew
                ind_rews.append(rep_rews)
            fitnesses.append(np.mean(ind_rews))
        population.fitnesses = np.array(fitnesses)
