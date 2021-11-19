import numpy as np
import argparse
import time

from multiagent.environment import MultiAgentEnv

from coop_push_scenario_closed import Scenario

class RandomPolicy():
    def __init__(self, env):
        self.env = env

    def action(self, obs):
        obj_pos = obs[16:18]
        print(obj_pos)
        if not obj_pos.any():
            return np.random.uniform(-1, 1, self.env.world.dim_p)
        else:
            dir_vec = 0.1 * obj_pos / np.sqrt(np.sum(np.square(obj_pos)) + 0.001)
            return dir_vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_render", action='store_true')
    config = parser.parse_args()

    seed = np.random.randint(1e9)
    np.random.seed(seed)

    scenario = Scenario()
    # Create world
    world = scenario.make_world(obs_range=1.0)

    # Create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    #env.render()

    # Create policies
    policies = [RandomPolicy(env) for i in range(env.n)]

    obs_n = env.reset()
    it = 0
    while True:
        # Get each agent's action
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
            print(f'Agent {i}: \nobs:{obs_n[i]}\naction:{act_n[i]}')

        # Environment step
        obs_n, reward_n, done_n, _ = env.step(act_n)
        for obs in obs_n:
            if np.isnan(obs).any():
                print("NAN IN OBS")
                print(obs_n)
                print("seed:", seed)
                exit(1)
        print(reward_n)

        time.sleep(0)
        if not config.no_render:
            env.render()
        it += 1
        if it == 200:
            print("seed:", seed)
            break