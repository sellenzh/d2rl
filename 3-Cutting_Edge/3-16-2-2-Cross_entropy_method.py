import numpy as np
from scipy.stats import truncnorm
import gym
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt

class CEM:
    def __init__(self, n_squence: int, 
                 elite_ratio: float, 
                 fake_env, 
                 upper_bound, 
                 lower_bound) -> None:
        self.n_squence = n_squence
        self.elite_ratio = elite_ratio
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.fake_env = fake_env

    def optimize(self, state: str,
                 init_mean: float,
                 init_var: float) -> float:
        mean, var = init_mean, init_var
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        state = np.tile(state, (self.n_squence, 1))

        for _ in range(5):
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            # generate action squence.
            action_squences = [X.rvs() for _ in range(self.n_squence)] * np.sqrt(constrained_var) + mean

            # calculate accumulated rewards for each action squence.
            returns = self.fake_env.propagate(state, action_squences)[:, 0]

            #s select some most accumulated rewards squence.
            elites = action_squences[np.argsort(returns)] [-int(self.elite_ratio * self.n_squence):]

            new_mean = np.mean(elites, axis = 0)
            new_var = np.var(elites, axis = 0)

            # update the distribution of action squence.
            mean = .1 * mean + .9 * new_mean
            var = .1 * var + .9 * new_var
        return mean
        