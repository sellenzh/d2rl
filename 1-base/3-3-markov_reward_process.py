"""
Markov Reward Process
25-04-21
"""
from typing import Optional, List, Dict, Tuple
import numpy as np

np.random.seed(42)

# State transfer matrix.
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

# rewards
rewards = [-1, -2, -2, 10, 1, 0]

# discount factor.
gamma = 0.5


def compute_return(start_index: int, chain: List[int], gamma: float) -> float:
    """
    in 'chain', return 'reward' from 'start_index' to end state.
    :param start_index: index of initial state.
    :param chain: sequence.
    :param gamma: discount factor.
    :return: reward.
    """
    G = .0
    for i in reversed(range(start_index, len(chain))):
        G = rewards[chain[i] - 1] + gamma * G
    return G


chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)

print(f'rewards: {G}')


def compute_bellman(P: List, rewards: List, gamma: float, states_num: int) -> float:
    """
    
    :param P: State transfer matrix.
    :param rewards: 
    :param gamma: discount factor.
    :param state_num:
    :return: reward.
    """
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


V = compute_bellman(P, rewards, gamma, states_num=6)
print(f'MRP value for each state: {V}')


# State set.
S = ['s1', 's2', 's3', 's4', 's5']

# Action set.
A = ['ks1', 'fs1', 'fs2', 'fs3', 'fs4', 'fs5', 'fp']

# State transfer matrix.
P = {
    's1-ks1-s1': 1.0,
    's1-fs2-s2': 1.0,
    's2-fs1-s1': 1.0,
    's2-fs3-s3': 1.0,
    's3-fs4-s4': 1.0,
    's3-fs5-s5': 1.0,
    's4-fs5-s5': 1.0,
    's4-fp-s2': 0.2,
    's4-fp-s3': 0.4,
    's4-fp-s4': 0.4,
}

# Reward
R = {
    's1-ks1': -1,
    's1-fs2': 0,
    's2-fs1': -1,
    's2-fs3': -2,
    's3-fs4':-2,
    's3-fs5': 0,
    's4-fs5': 10,
    's4-fp': 1,
}

# discount factor
gamma = 0.5

MDP = (S, A, P, R, gamma)

Pi = {
    's1-ks1': .5,
    's1-fs2': .5,
    's2-fs1': .5,
    's2-fs3': .5,
    's3-fs4': .5,
    's3-fs5': .5,
    's4-fs5': .5,
    's4-fp': .5,
}

Pi_2 = {
    's1-ks1': .6,
    's1-fs2': .4,
    's2-fs1': .3,
    's2-fs3': .7,
    's3-fs4': .5,
    's3-fs5': .5,
    's4-fs5': .1,
    's4-fp': .9,
}


def join(str1: str, str2: str) -> str:
    return str1 + '-' + str2


def sample_monte_carlo(MDP: Tuple, Pi: Dict, time_step_max: int, number: int) -> List[Tuple]:
    """
    sample use monte carlo method.
    :param MDP:
    :param Pi: policy.
    :param time_step_max: max sample time step.
    :param number: sample number of sequence.
    :return: 
    """
    S, A, P, R, gamma = MDP
    episodes = []
    
    for _ in range(number):
        episode = []
        time_step = 0
        
        # random choice a state (except s5) as initial state.
        s = S[np.random.randint(4)]
        while s != 's5' and time_step <= time_step_max:
            time_step += 1
            rand, temp = np.random.rand(), 0

            # select action under state 's'
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            
            rand, temp = np.random.rand(), 0

            # get next state with state transfer matrix.
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes

# episodes = sample_monte_carlo(MDP, Pi, time_step_max=20, number=5)

# print(f'1st sequence: {episodes[0]}')
# print(f'2nd sequence: {episodes[1]}')
# print(f'5th sequence: {episodes[4]}')


def MC(episodes: List, V: Dict, N: Dict, gamma: float) -> None:
    """
    
    """
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1): # a len from back to head
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]



episodes = sample_monte_carlo(MDP, Pi, time_step_max=20, number=1000)

gamma = .5
V = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
N = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
MC(episodes, V, N, gamma)
print(f'state value with monte carlo: {V}')



def occupancy(episodes: List, s: Dict, a: Dict, time_step_max: int, gamma: float):
    """
    
    """
    rho = 0
    total_times = np.zeros(time_step_max)
    occur_times = np.zeros(time_step_max)
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(time_step_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = .5
time_step_max = 1000

episodes_1 = sample_monte_carlo(MDP, Pi, time_step_max, 1000)
episodes_2 = sample_monte_carlo(MDP, Pi_2, time_step_max, 1000)
rho_1 = occupancy(episodes_1, 's4', 'fp', time_step_max, gamma)
rho_2 = occupancy(episodes_2, 's4', 'fp', time_step_max, gamma)

print(rho_1, rho_2)
