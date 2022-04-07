import numpy as np

from .runner import Runner


class RNNRunner(Runner):

    def __init__(self, args, num_agents, policy, env, buffer):
        """
        Runner for model using RNN layers.
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

        rnn_states_batch = np.zeros(
            (self.args.n_rollout_threads * self.num_agents,
             self.args.hidden_size), 
            dtype=np.float32)
        last_acts_batch = np.zeros(
            (self.args.n_rollout_threads * self.num_agents, self.policy.output_dim), 
            dtype=np.float32)
        episode_obs = {"policy_0" : np.zeros((self.args.episode_length + 1, 
                            self.args.n_rollout_threads, 
                            self.num_agents, 
                            self.policy.obs_dim), 
                            dtype=np.float32)}
        episode_share_obs = {"policy_0" : np.zeros(
            (self.args.episode_length + 1, 
             self.args.n_rollout_threads, 
             self.num_agents, 
             self.policy.central_obs_dim), dtype=np.float32)}
        episode_acts = {"policy_0" : np.zeros(
            (self.args.episode_length, 
             self.args.n_rollout_threads, 
             self.num_agents, 
             self.policy.output_dim), dtype=np.float32)}
        episode_rewards = {"policy_0" : np.zeros(
            (self.args.episode_length, 
             self.args.n_rollout_threads, 
             self.num_agents, 
             1), dtype=np.float32)}
        episode_dones = {"policy_0" : np.ones(
            (self.args.episode_length, 
             self.args.n_rollout_threads, 
             self.num_agents, 
             1), dtype=np.float32)}
        episode_dones_env = {"policy_0" : np.ones(
            (self.args.episode_length, 
             self.args.n_rollout_threads, 
             1), dtype=np.float32)}
        episode_avail_acts = {"policy_0" : None}
        # Log info
        ep_dones = np.zeros(self.args.n_rollout_threads)
        ep_length = np.ones(self.args.n_rollout_threads) * self.args.episode_length
        for step_i in range(self.args.episode_length):
            share_obs = obs.reshape(self.args.n_rollout_threads, -1)
            # Copy over agent dim
            share_obs = np.repeat(share_obs[:, np.newaxis, :], 
                                  self.num_agents, axis=1)
            # group observations from parallel envs into one batch to process
            # at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env with exploration noise
            acts_batch, rnn_states_batch, _ = self.policy.get_actions(
                                                    obs_batch,
                                                    last_acts_batch,
                                                    rnn_states_batch,
                                                    t_env=ep_i,
                                                    explore=True)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else\
                            acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch \
                if isinstance(rnn_states_batch, np.ndarray) \
                else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch
            # print("ACTIONS 1", acts_batch)
            env_acts = np.split(acts_batch, self.args.n_rollout_threads)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = self.env.step(env_acts)
            # print("obs", obs_batch)
            # print("actions", acts_batch)
            # print("rewards", rewards)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or \
                                 step_i == self.args.episode_length - 1

            episode_obs["policy_0"][step_i] = obs
            episode_share_obs["policy_0"][step_i] = share_obs
            episode_acts["policy_0"][step_i] = np.stack(env_acts)
            episode_rewards["policy_0"][step_i] = rewards[..., np.newaxis]
            episode_dones["policy_0"][step_i] = dones[..., np.newaxis]
            episode_dones_env["policy_0"][step_i] = dones_env[..., np.newaxis]

            # Check for dones in each rollout
            for r_i in range(self.args.n_rollout_threads):
                if ep_dones[r_i] == 0 and dones[r_i][0]:
                    ep_dones[r_i] = 1
                    ep_length[r_i] = step_i
            if dones.sum(1).all():
                # print(f"Finished early! (step {ep_length}, reward {np.sum(episode_rewards["policy_0"], axis=0)})")
                break

            obs = next_obs
            # env.render()
            if terminate_episodes:
                break
        
        episode_obs["policy_0"][step_i + 1] = obs
        share_obs = obs.reshape(self.args.n_rollout_threads, -1)
        share_obs = np.repeat(share_obs[:, np.newaxis, :], self.num_agents, axis=1)
        episode_share_obs["policy_0"][step_i + 1] = share_obs

        # Save returns on this round of rollouts
        episode_return = np.sum(episode_rewards["policy_0"],axis=0)
        # print(episode_return)

        # push all episodes collected in this rollout step to the buffer
        self.buffer.insert(
            self.args.n_rollout_threads,
            episode_obs,
            episode_share_obs,
            episode_acts,
            episode_rewards,
            episode_dones,
            episode_dones_env,
            episode_avail_acts)

        return episode_return, ep_dones, ep_length

    def eval_rollout(self):
        pass
