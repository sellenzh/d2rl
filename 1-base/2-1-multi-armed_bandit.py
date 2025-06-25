"""
multi armed bandit problem.
2025-06-25
"""

import numpy as np
# from matplotlib.pyplot as plt

seed = 42

class BernoulliBandit:
    def __init__(self, K=int) -> None:
        '''
        :param K: (int) number of bandit.
        '''
        self.probs = np.random.uniform(size=K) # generate K probs between 0~1 as reward in bandit.
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k: int) -> int:
        '''
        :param k: (int) choose bandit.
        :return: (int) reward.
        '''
        return 1 if np.random.rand() < self.probs[k] else 0

np.random.seed(seed)
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f'random generate {K} bernoulli armed bandit.')
print(f'获奖概率最大的拉杆为{bandit_10_arm.best_idx}号，其获奖概率为{bandit_10_arm.best_prob:.4f}')

# ==========*===========*========== #

class Solver:
    '''
    multi armed bandit base framework.
    1. choose action based on policy.
    2. get reward based on action.
    3. update expected reward estimate.
    4. update cumulative regret and count.
    '''
    def __init__(self, bandit: BernoulliBandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数
        self.regret = 0. # 当前 step 积累的懊悔
        self.actions = [] # action list.
        self.regrets = [] # regrets list.

    def update_regret(self, k: int) -> None:
        # calculate cumulative regret. 'k' is number of bandit based choosed action.
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self) -> None:
        # return which bandit based on action, implement by policy.
        raise NotImplementedError
    
    def run(self, num_steps: int) -> None:
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

