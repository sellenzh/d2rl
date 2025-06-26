import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time
from typing import List


class CliffWalkingEnv:
    def __init__(self, ncol: int, nrow: int) -> None:
        self.ncol, self.nrow = ncol, nrow
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action: int) -> tuple:
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done
    
    def reset(self) -> int:
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    

class DynaQ:
    def __init__(self, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float, n_planning: int, n_action: int=4) -> None:
        self.Q_table = np.zeros([nrow * ncol, n_action]) # init Q table
        self.n_action = n_action
        self.alpha, self.epsilon, self.gamma = alpha, epsilon, gamma
        self.n_planning = n_planning
        self.model = dict()

    def take_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            # random choose an action based epsilon to explore
            action = np.random.randint(self.n_action) 
        else:
            # choose a kown action
            action = np.argmax(self.Q_table[state])
        return action
    
    def q_learning(self, s0: int, a0: int, r: float, s1: int) -> None:
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0: int, a0: int, r: float, s1: int) -> None:
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1
        for _ in range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


def DynaQ_CliffWalking(n_planning: int) -> List:
    ncol, nrow = 12, 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon, alpha, gamma = .01, .1, .9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        f'{num_episodes / 10 * i + i_episode + 1}',
                        'return':
                        f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
    return return_list


np.random.seed(0)
random.seed(0)
n_planning_list = [0, 2, 20]
for n_planning in n_planning_list:
    print(f'Q-planning步数为： {n_planning}')
    time.sleep(0.5)
    return_list = DynaQ_CliffWalking(n_planning)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,
             return_list,
             label=str(n_planning) + ' planning steps')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'Dyna-Q on {'Cliff Walking'}')
plt.savefig('Dyna_Q.jpg')