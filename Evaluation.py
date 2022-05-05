import numpy as np


class RewardMaximization():
    def __init__(self, env, model, reps=3):
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

