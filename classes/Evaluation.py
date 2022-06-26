import numpy as np
import torch

from EA_components import Population


class RewardMaximizationNN():
    def __init__(self, env, model, reps=3, render=False):
        self.env = env
        self.model = model
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
                    if self.env.action_space.__class__.__name__ == "Discrete":
                        a = np.argmax(self.model(state).numpy())
                    elif self.env.action_space.__class__.__name__ == "Box":
                        a = self.model(state).numpy()
                    else:
                        exit(f"{self.env.action_space.__class__.__name__} action space not yet implemented")
                    # query environment
                    state, rew, done, _ = self.env.step(a)
                    state = torch.tensor(state, requires_grad=False)
                    if self.render:
                        self.env.render()
                    rep_rews += rew
                ind_rews.append(rep_rews)
            fitnesses.append(np.mean(ind_rews))
        population.fitnesses = np.array(fitnesses)
