import numpy as np
import random
import matplotlib.pyplot as plt



class Action:
    def __init__(self, optimistic=False):
        self.Q = 10 if optimistic else 0
        self.n = 0


class Action:
    def __init__(self, optimistic=False):
        self.Q = 10 if optimistic else 0
        self.n = 0

class MultiArmsBanditAgent:
    def __init__(self, env):
        self.actions = [Action(optimistic=False) for _ in range(env.action_space.n)]
        
    def epsilone_greedy(self, epsilon):
        if random.uniform(0, 1) < epsilon: 
            return random.choice(list(range(len(self.actions))))
        else:  
            return self.greedy()

    def greedy(self):
        best_action = 0
        highest_q = self.actions[0].Q
        for i in range(1, len(self.actions)):
            current_q = self.actions[i].Q
            if current_q > highest_q:
                highest_q = current_q
                best_action = i
        return best_action

    def update(self, action, reward, alpha=None):
        self.actions[action].n += 1
        step_size = (1 / self.actions[action].n)
        
        if alpha is not None:
            step_size = alpha
            
        self.actions[action].Q = self.actions[action].Q + step_size * (reward - self.actions[action].Q)
        

class GradientBanditAgent(MultiArmsBanditAgent):
    def __init__(self, env, alpha):
        super().__init__(env)
        self.alpha = alpha
        self.H_values = [0] * len(self.actions)
        self.actions_probs = None

    def _cal_action_prob(self, h):
        return np.exp(h) / np.sum(np.exp(self.H_values))

    def select_action(self):
        self.action_probs = list(map(self._cal_action_prob, self.H_values))
        action = random.choices(list(range(len(self.actions))), self.action_probs, k=1)[0]
        return action

    def _update_preference(self, action, reward):
        reward_hat = self.actions[action].Q
        error = reward - reward_hat

        for a in list(range(len(self.actions))):
            if a != action:
                self._update_action_preference(a, error)
            else:
                self._update_action_preference(a, error, is_curr_action=True)

    def _update_action_preference(self, action, error, is_curr_action=False):
        action_pref = self.H_values[action]
        action_prob = self.action_probs[action]
        
        if is_curr_action:
            self.H_values[action] = action_pref + self.alpha * error * (1 - action_prob)
        else:
            self.H_values[action] = action_pref - self.alpha * error * action_prob



