"""
25-04-22
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

class CliffWalkingEnv:
    def __init__(self, ncol: int, nrow: int):
        self.ncol, self.nrow = ncol, nrow
        self.x, self.y = 0, self.nrow - 1

    def step(self, action: str) -> Tuple:
        # 4 actions: change[0]: up, change[1]: down, change[2]: left, change[3]: right.
        # define init location: up-left.
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
    
    def reset(self,):
         self.x = 0
         self.y = self.nrow - 1
         return self.y * self.ncol + self.x
    

class Sarsa:
    def __init__(self, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float, n_action: int = 4):
        # init Q table
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state: str) -> str:
        return np.random.randint(self.n_action) if np.random.random() < self.epsilon else np.argmax(self.Q_table[state])
    
    def best_action(self, state: str) -> List:
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] == 1
        return a
    
    def update(self, s0: str, a0: str, r: float, s1: str, a1: str) -> None:
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

        
# ncol, nrow = 12, 4
# env = CliffWalkingEnv(ncol, nrow)

# np.random.seed(0)

# epsilon, alpha, gamma = .1, .1, .9

# agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
# num_episodes = 500

# return_list = []
# for i in range(10):
#     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
#         for i_episode in range(int(num_episodes / 10)):
#             episode_return = 0
#             state = env.reset()
#             action = agent.take_action(state)
#             done = False

#             while not done:
#                 next_state, reward, done = env.step(action)
#                 next_action = agent.take_action(next_state)
#                 episode_return += reward 
#                 agent.update(state, action, reward, next_state, next_action)
#                 state = next_state
#                 action = next_action
            
#             return_list.append(episode_return)
#             if (i_episode + 1) % 10 == 0:
#                 pbar.set_postfix({
#                     'episode': f'{num_episodes / 10 * i + i_episode + 1}',
#                     'return': f'{np.mean(return_list[-10:])}'
#                 })
#             pbar.update(1)

# episode_list = list(range(len(return_list)))
# plt.plot(episode_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Sarsa on {}'.format('Cliff Walking'))
# plt.savefig('saras.jpg')
        

def print_agent(agent: Sarsa, env: CliffWalkingEnv, action_meaning: str, disaster: List = [], end: List = []):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.nrow + j) in end:
                print('EEEE', end=' ') 
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


# action_meaning = ['^', 'v', '<', '>']
# print('Convergence Policy with Sarsa: ')
# print_agent(agent, env, action_meaning, list(range(37, 47)), [47])


class nstep_Sarsa:
    def __init__(self, n: int, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float, n_action: int = 4) -> None:
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.state_list = []
        self.action_list = []
        self.reward_list = []

    def take_action(self, state: str) -> str:
        return np.random.randint(self.n_action) if np.random.random() < self.epsilon else np.argmax(self.Q_table[state])

    def best_action(self, state: str) -> str:
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
    
    def update(self, s0: str, a0: str, r: float, s1: str, a1: str, done: bool) -> None:
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        if len(self.state_list) == self.n: # save data support n step update.
            G = self.Q_table[s1, a1]
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i] # calculate forward each step reward.
                # if touch terminal state, though steps less than n also update.
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            # delete outdated state from memory list and not update next time.
            s = self.state_list.pop(0) 
            a = self.action_list.pop(0)
            self.reward_list.pop(0)

            # n step sarsa main update procedure.
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []


np.random.seed(0)
n_step = 5
ncol, nrow = 12, 4
env = CliffWalkingEnv(ncol, nrow)

alpha, epsilon, gamma = .1, .1, .9
agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500


return_list = []
for i in range(10):
    with tqdm(total= int(num_episodes / 10), desc=f'Iteration: {i}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False

            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward
                agent.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                    'return': f'{np.mean(return_list[-10:])}'
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
plt.savefig('5-step-sarsa.jpg')
        