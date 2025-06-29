
import random
import gym
import numpy as np
import collections

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class ReplayBuffer:
    '''
    经验放回池
    '''
    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: dict, action: int, reward: int, next_state: int, done: bool) -> None:
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self) -> int:
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    '''
    只有一层隐藏层的Q网络
    '''
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class DQN:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 learning_rate: float, gamma: float, epsilon: float, target_update: float, 
                 device: str) -> None:
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # Q 网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device) # 目标网络

        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma, self.epsilon, self.target_update = gamma, epsilon, target_update
        self.count = 0
        self.device = device

    def take_action(self, state: dict) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self, transition_dict: dict) -> None:
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions) # Q 值
        # next state max value
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # td 误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # mse loss
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = .98
epsilon = .01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
seed = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env_name = 'CartPole-v1'
env = gym.make(env_name)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()[0]
            done = False
            steps = 0
            while not done and steps < 200:
                steps += 1
                action = agent.take_action(state)
                next_state, reward, done, _, __ = env.step(action)
                replay_buffer.add(state, int(action), reward, next_state, done)
                state = next_state
                episode_return += reward

                # 当 buffer 数据的数量超过一定值之后，才进行 Q 网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                    'return': f'{np.mean(return_list[-10:]):.3f}'
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('DQN.jpg')