"""
chapter 4 sample. Cliff Walking.
25-04-22
"""

import copy
from typing import List, Dict

class CliffWalkingEnv():
    """
    create cliff walking enviroment.
    """
    def __init__(self, n_col: int = 12,    # the column of the  Grid world.
                    n_row: int = 4,     # the row of the Grid world.
        ) -> None:
        self.n_col = n_col
        self.n_row = n_row

        # state transfer matrix P[state][action] = [(p, next_state, reward, done)] include next state and reward.
        self.P = self.createP()

    def createP(self,) -> List:
        # init
        P = [[[] for j in range(4)] for i in range(self.n_row * self.n_col)]

        # 4 actions, change[0]: up, change[1]: down, change[2]: left, change[3]: right. initial location: [0, 0].
        # define initial locate at up-left.
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.n_row):
            for j in range(self.n_col):
                for a in range(4):
                    # location at cliff or target state, cause not to interact any action reward is 0.
                    if i == self.n_row - 1 and j > 0:
                        P[i * self.n_col + j][a] = [(1, i * self.n_col + j, 0, True)]
                        continue
                    # other location
                    next_x = min(self.n_col - 1, max(0, j + change[a][0]))
                    next_y = min(self.n_row - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.n_col + next_x
                    reward = -1
                    done = False

                    # next location is terminal or cliff
                    if next_y == self.n_row - 1 and next_x > 0:
                        done = True
                        if next_x != self.n_col - 1: # next location at cliff.
                            reward = -100
                    P[i * self.n_col + j][a] = [(1, next_state, reward, done)]
        return P
    

class PolicyIteration:
    """
    Policy iteratyion algorithm.
    """
    def __init__(self, env: CliffWalkingEnv, theta: float, gamma: float) -> None:
        self.env = env
        self.v = [0] * self.env.n_col * self.env.n_row # initial 0 for each values.
        self.pi = [[.25, .25, .25, .25] for i in range(self.env.n_col * self.env.n_row)] # init uniform random policy.

        self.theta = theta # policy evaluation convergence threshold
        self.gamma = gamma # discount factor
    
    def policy_evaluation(self,) -> None:
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * self.env.n_col * self.env.n_row
            for s in range(self.env.n_col * self.env.n_row):
                qsa_list = [] # calculate each Q(s,a) under state s.
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        
                        # rewards is related to the next state.
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list) # relationship bewteen state value function and action value function.
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break # satisfied convergence condition, quit evalute iteration.
            cnt += 1
        print(f'policy evaluation finished after {cnt} turns.')

    
    def policy_improvement(self) -> List:
        for s in range(self.env.n_row * self.env.n_col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # calculate how many action's value get max value.
            
            # get these actions uniform the possibility.
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print('policy improvement finished.')
        return self.pi

    def policy_iteration(self) -> None:
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi) # deep copy the list to compare.
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break


def print_agent(agent,
                action_meaning,
                disaster=[],
                end=[]):
    print('state value: ')
    for i in range(agent.env.n_row):
        for j in range(agent.env.n_col):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.n_col + j]), end=' ')
        print()

    print('policy: ')
    for i in range(agent.env.n_row):
        for j in range(agent.env.n_col):
            if (i * agent.env.n_col + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.n_col + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.n_col + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = .001
gamma = .9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])



class ValueIteration:
    def __init__(self, env: CliffWalkingEnv, theta: float, gamma: float) -> None:
        self.env = env
        self.theta = theta
        self.gamma = gamma

        self.v = [0] * self.env.n_col * self.env.n_row
        self.pi = [None for i in range(self.env.n_col * self.env.n_row)]

    def value_iteration(self) -> None:
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.n_col * self.env.n_row
            for s in range(self.env.n_col * self.env.n_row):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
            cnt += 1
        print(f'value iteration run {cnt} turns.')
        self.get_policy()

    def get_policy(self,) -> None:
        for s in range(self.env.n_row * self.env.n_col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)

            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = .001
gamma = .9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

import gym

env = gym.make('FrozenLake-v1', render_mode='rgb_array')
env = env.unwrapped


holes = set()
ends = set()

for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.:
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])

holes = holes - ends
print(f'index of frozen lake: {holes}')
print(f'index of target: {ends}')

for a in env.P[14]:
    print(env.P[14][a])

# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])