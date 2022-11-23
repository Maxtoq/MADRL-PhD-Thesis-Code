import numpy as np
import random

BEST_REWARD = 12
MID_REWARD = 0

class RelOvergenEnv:

    def __init__(self, state_dim):
        self.obs_dim = state_dim
        self.act_dim = 2

        self.state_dim = state_dim
        self.unit = 10.0 / state_dim
        self.states = list(np.arange(0.0, 10.0, self.unit))

        self.agents_pos = [0, 0]

        self.optimal_state = [
            int(state_dim / 4) * self.unit, 
            int(state_dim / 5) * self.unit]
        self.suboptimal_state = [
            10.0 - int(state_dim / 4) * self.unit, 
            10.0 - int(state_dim / 5) * self.unit]
        
        self.max_steps = state_dim
        self.current_step = 0

    def get_obs(self):
        return [
            np.eye(self.state_dim)[self.agents_pos[0]],
            np.eye(self.state_dim)[self.agents_pos[1]]
        ]

    def reset(self):
        for a_i in range(2):
            self.agents_pos[a_i] = random.randint(0, self.state_dim - 1)
        self.current_step = 0
        return self.get_obs()

    def compute_reward(self):
        opti = BEST_REWARD - 25 * (
            (self.states[self.agents_pos[0]] - self.optimal_state[0]) ** 2 + 
            (self.states[self.agents_pos[1]] - self.optimal_state[1]) ** 2)
        subopti = MID_REWARD - 0.125 * (
            (self.states[self.agents_pos[0]] - self.suboptimal_state[0]) ** 2 + 
            (self.states[self.agents_pos[1]] - self.suboptimal_state[1]) ** 2)
        return max(opti, subopti)

    def step(self, actions):
        for a_i in range(2):
            self.agents_pos[a_i] += int(actions[a_i][0])
            self.agents_pos[a_i] -= int(actions[a_i][1])
            if self.agents_pos[a_i] < 0:
                self.agents_pos[a_i] = 0
            elif self.agents_pos[a_i] >= self.state_dim:
                self.agents_pos[a_i] = self.state_dim - 1
        next_states = self.get_obs()

        reward = self.compute_reward()
        rewards =[reward, reward]
        
        self.current_step += 1
        done = float(self.current_step >= self.max_steps)
        dones = [done, done]
        
        return next_states, rewards, dones, None
    
    def close(self):
        pass