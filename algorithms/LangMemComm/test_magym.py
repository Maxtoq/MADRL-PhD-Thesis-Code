import gym
import numpy as np


class KeyboardMAS():

    possible_actions = "sqzd "

    def __init__(self, n_agents):
        self.n_agents = n_agents

    def _str_actions_valid(self, str_actions):
        if (len(str_actions) != self.n_agents or 
                not all([str_actions[i] in self.possible_actions 
                         for i in range(len(str_actions))])):
            return False
        else:
            return True

    def _str_actions_to_numpy(self, str_actions):
        actions = [self.possible_actions.find(str_actions[i]) 
                   for i in range(self.n_agents)]
        return np.array(actions)

    def get_actions(self):
        while True:
            str_actions = input(f"Enter {self.n_agents} actions (valid actions are '{self.possible_actions}'):")
            if self._str_actions_valid(str_actions):
                break
            else:
                print("Invalid actions")
        return self._str_actions_to_numpy(str_actions)



gym.envs.register(
    id='PredPrey5x5-v0',
    entry_point='src.envs.ma_gym.envs.predator_prey:PredatorPrey'
    #kwargs={'n_agents': 2, 'full_observable': False, 'step_cost': -0.2} 
    # It has a step cost of -0.2 now
)

if __name__ == "__main__":
    env = gym.make('PredPrey5x5-v0')

    actor = KeyboardMAS(env.n_agents)

    dones = [False for _ in range(env.n_agents)]
    ep_reward = 0

    obs = env.reset()
    while not all(dones):
        env.render()
        print("Observations:")
        for a_i in range(env.n_agents):
            print(f"A{a_i}:", len(obs[a_i]), obs[a_i])
        actions = actor.get_actions()
        #env.action_space.sample()
        obs, rewards, dones, infos = env.step(actions)
        print("Actions:", actions)
        print("Rewards", rewards)
        ep_reward += sum(rewards)
        # time.sleep(0.01)
    env.close()