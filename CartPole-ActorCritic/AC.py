import numpy as np
import torch.nn as nn
import torch
from collections import deque
from collections import defaultdict

class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.Softmax(dim=1)(x)
    
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
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def A2C(policy, value_net, env, optimizer_policy, optimizer_value, n_episodes=2000, max_t=1000, gamma=0.99, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []

    for e in range(1, n_episodes + 1):
        saved_log_probs = []
        values = []
        rewards = []
        state = env.reset()

        if isinstance(state, tuple):
            state = state[0]
        
        for t in range(max_t):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device="cpu")
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            values.append(value_net(state_tensor).item())
            
            if done:
                break

            state = next_state
        
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(device="cpu")
        next_value = value_net(next_state_tensor).item()
        values.append(next_value)

        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + gamma * R
            returns[t] = R
            advantages[t] = R - values[t]
        
        returns = torch.tensor(returns, dtype=torch.float).to(device="cpu")
        saved_log_probs = torch.stack(saved_log_probs).to(device="cpu")
        advantages = torch.tensor(advantages, dtype=torch.float).to(device="cpu")
        
        policy_loss = -torch.sum(saved_log_probs * advantages)

        value_loss = torch.nn.MSELoss()(torch.tensor(values[:-1], dtype=torch.float).to(device="cpu"), returns[:-1])
        
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
        
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        if e % print_every == 0:
            print(f"Episode {e}\tAverage Score: {np.mean(scores_deque)}")