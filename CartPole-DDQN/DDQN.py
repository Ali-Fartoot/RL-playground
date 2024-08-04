import gymnasium as gym
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim
import torch
import collections
import random
from collections import namedtuple
import numpy as np

class FCModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 16)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(16,16)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(16,16)
        self.activation3 = nn.ReLU()

        self.output_layer = nn.Linear(16, output_size)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x

class QNetwork:
    def __init__(self, env, lr, logdir=None):
        self.net = FCModel(4,2)
        self.env = env
        self.lr = lr
        self.logdir = logdir
        self.optimizer = optim.Adam(self.net.parameters(),self.lr)

    def load_model(self, model_file):
        return self.net.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        return self.net.load_state_dict(torch.load(weight_file))

class Memory:
    def __init__(self, env, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = collections.deque([], maxlen=memory_size)
        self.env = env

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        self.memory.append(transition)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DDQNAgent:
    def __init__(self, environment_name, lr=5e-4, render=False):
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.lr = lr
        self.policy_net = QNetwork(self.env, self.lr)
        self.target_net = QNetwork(self.env, self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())
        self.rm = Memory(self.env)
        self.burn_in_memory()
        self.batch_size = 32
        self.gamma = 0.99
        self.c = 0

    def burn_in_memory(self):
        cnt = 0
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while cnt < self.rm.burn_in:
            if terminated or truncated:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            action = torch.tensor(random.sample([0, 1], 1)[0]).reshape(1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
            transition = Transition(state, action, next_state, reward)
            self.rm.append(transition)
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def greedy_policy(self, q_values):
        return torch.argmax(q_values)
        
    def train(self):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.policy_net.net(state)

            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            transition = Transition(state, action, next_state, reward)
            self.rm.append(transition)
            state = next_state

            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.policy_net.net(state_batch).gather(1, action_batch)

            with torch.no_grad():
                next_state_actions = self.policy_net.net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values = torch.zeros(self.batch_size)
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).gather(1, next_state_actions).squeeze(1)

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

    def test(self, model_file=None):
        max_t = 1000
        state, _ = self.env.reset()
        rewards = []

        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break

        return np.sum(rewards)

