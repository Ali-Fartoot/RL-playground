import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return nn.Softmax(dim=1)(self.fc2(x))
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device="cpu")
        probs = self.forward(state).cpu()
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class ValueNetwork(nn.Module):
    def __init__(self, state_size=4, hidden_size=32):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def A2C(policy, value_net, env, optimizer_policy, optimizer_value, n_episodes=500, max_t=1000, gamma=0.99, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []

    for e in range(1, n_episodes + 1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        saved_log_probs = []
        values = []
        rewards = []
        
        for t in range(max_t):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action, log_prob = policy.act(state)
            value = value_net(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            saved_log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns)
        values = torch.cat(values)
        
        advantages = returns - values.detach()
        
        policy_loss = -(torch.stack(saved_log_probs) * advantages).mean()
        value_loss = nn.MSELoss()(values, returns)
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
        
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        if e % print_every == 0:
            print(f"Episode {e}\tAverage Score: {np.mean(scores_deque):.2f}")
    
    return scores
