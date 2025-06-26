"""
multi armed bandit problem.
2025-06-25
"""

import numpy as np
import matplotlib.pyplot as plt

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

class EpsilonGreedy(Solver):
    def __init__(self, bandit: int, epsilon: float=.01, init_prob: float=1.) -> None:
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # init 所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self) -> int:
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # random choose a bandit
        else:
            k = np.argmax(self.estimates) # choose max reward bandit
        r = self.bandit.step(k) # get reward for the bandit.
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers: Solver, solver_names: str) -> None:
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title(f'{solvers[0].bandit.K}-armed bandit')
    plt.legend()
    plt.savefig(f'{solver_names}.jpg')


np.random.seed(seed)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=.01)
epsilon_greedy_solver.run(5000)
print(f'epsilon-greedy 的累积懊悔值为：{epsilon_greedy_solver.regret}')
plot_results([epsilon_greedy_solver], ['EpsilonGreedy'])

np.random.seed(0)
epsilons = [1e-4, .01, .1, .25, .5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = [f'epsilon={e}' for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


# ==========*===========*========== #
# epsilon 随时间衰减
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit: int, init_prob: float=1.) -> None:
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self) -> int:
        self.total_count += 1
        if np.random.random() < 1 / self.total_count: # epsilon 随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(seed)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)

print(f'epsilon衰减的贪婪算法的累积懊悔为：{decaying_epsilon_greedy_solver.regret}')
plot_results([decaying_epsilon_greedy_solver], ['DecayingEpsilonGreedy'])

# ==========*===========*========== #
# 上置信界算法
class UCB(Solver):
    def __init__(self, bandit: int, coef: float, init_prob: float=1.) -> None:
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self) -> int:
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(seed)
coef = 1
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])

# ==========*===========*========== #
# 汤普森采样法
class ThompsonSampling(Solver):
    def __init__(self, bandit: int) -> None:
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self) -> int:
        samples = np.random.beta(self._a, self._b) # 按照 beta 分布采样一组奖励样本
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1 - r)
        return k
    
np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])