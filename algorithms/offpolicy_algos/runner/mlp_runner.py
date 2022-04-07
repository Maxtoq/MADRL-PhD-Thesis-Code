import numpy as np

from .runner import Runner


class MLPRunner(Runner):

    def __init__(self, args, num_agents, policy, env, buffer):
        """
        Runner for model using MLP layers.
        :param args: arguments passed to the script
        :param num_agents: number of agents
        :param policy: model for outputing actions
        :param env: environment
        :param buffer: replay buffer
        """
        self.args = args
        self.num_agents = num_agents
        self.policy = policy
        self.env = env
        self.buffer = buffer

    def train_rollout(self, ep_i):
        """
        Rollouts a training episode.
        :param ep_i: training iteration
        """
        obs = self.env.reset()
        share_obs = obs.reshape(self.args.n_rollout_threads, -1)
        
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}
        # Log info
        ep_returns = np.zeros(self.args.n_rollout_threads)
        ep_dones = np.zeros(self.args.n_rollout_threads)
        ep_length = np.ones(self.args.n_rollout_threads) * self.args.episode_length
        for step_i in range(self.args.episode_length):
            # Copy over agent dim
            share_obs = np.repeat(share_obs[:, np.newaxis, :], 
                                  self.num_agents, axis=1)
            # group observations from parallel envs into one batch to process
            # at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env with exploration noise
            acts_batch, _ = self.policy.get_actions(
                                obs_batch,
                                t_env=ep_i,
                                explore=True)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else\
                            acts_batch.cpu().detach().numpy()
            # print("ACTIONS 1", acts_batch)
            env_acts = np.split(acts_batch, self.args.n_rollout_threads)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = self.env.step(env_acts)
            # print("obs", obs_batch)
            # print("actions", acts_batch)
            # print("rewards", rewards)
            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or \
                                 step_i == self.args.episode_length - 1

            next_share_obs = next_obs.reshape(self.args.n_rollout_threads, -1)

            step_obs["policy_0"] = obs
            step_share_obs["policy_0"] = share_obs
            step_acts["policy_0"] = env_acts
            step_rewards["policy_0"] = rewards[..., np.newaxis]
            step_next_obs["policy_0"] = next_obs
            step_next_share_obs["policy_0"] = next_share_obs
            step_dones["policy_0"] = np.zeros_like(dones)[..., np.newaxis]
            step_dones_env["policy_0"] = dones_env[..., np.newaxis]
            valid_transition["policy_0"] = np.ones_like(dones)[..., np.newaxis]
            step_avail_acts["policy_0"] = None
            step_next_avail_acts["policy_0"] = None

            # Save in replay buffer
            self.buffer.insert(
                self.args.n_rollout_threads,
                step_obs,
                step_share_obs,
                step_acts,
                step_rewards,
                step_next_obs,
                step_next_share_obs,
                step_dones,
                step_dones_env,
                valid_transition,
                step_avail_acts,
                step_next_avail_acts)

            # Check for dones in each rollout
            for r_i in range(self.args.n_rollout_threads):
                if ep_dones[r_i] == 0 and dones[r_i][0]:
                    ep_dones[r_i] = 1
                    ep_length[r_i] = step_i
            if dones.sum(1).all():
                # print(f"Finished early! (step {ep_length}, reward {np.sum(episode_rewards["policy_0"], axis=0)})")
                break
            if terminate_episodes:
                break

            obs = next_obs
            share_obs = next_share_obs
            # env.render()

        # Save returns on this round of rollouts
        episode_return = np.mean(np.sum(episode_rewards,axis=0), axis=1)

        return episode_return, ep_dones, ep_length

    def eval_rollout(self):
        pass
