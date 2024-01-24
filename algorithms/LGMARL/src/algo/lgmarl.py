import torch
import numpy as np

from .language.lang_learner import LanguageLearner
from .policy.acc_mappo import ACC_MAPPO


class LanguageGroundedMARL:

    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 vocab, device="cpu", comm_logger=None):
        self.args = args
        self.n_agents = n_agents
        self.context_dim = args.context_dim
        self.n_envs = args.n_parallel_envs
        self.n_warmup_steps = args.n_warmup_steps
        self.token_penalty = args.comm_token_penalty
        self.env_reward_coef = args.comm_env_reward_coef
        self.comm_type = args.comm_type
        self.comm_ec_strategy = args.comm_ec_strategy
        self.comm_logger = comm_logger
        self.device = device

        # Modules
        self.lang_learner = LanguageLearner(
            obs_space[0].shape[0], 
            self.context_dim, 
            args.lang_hidden_dim, 
            vocab, 
            device,
            args.lang_lr,
            args.lang_n_epochs,
            args.lang_batch_size)

        self.comm_n_act_policy = ACC_MAPPO(
            args, 
            self.lang_learner, 
            n_agents, 
            obs_space, 
            shared_obs_space, 
            act_space[0], 
            device)

        # Language context, to carry to next steps
        self.lang_contexts = None
        # Matrices used during rollout
        self.values = None
        self.actions = None
        self.action_log_probs = None
        self.comm_actions = None
        self.comm_action_log_probs = None
        self.rnn_states = None
        self.rnn_states_critic = None

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        self.lang_learner.prep_training(self.device)
        self.comm_n_act_policy.prep_training(self.device)

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        self.lang_learner.prep_rollout(self.device)
        self.comm_n_act_policy.prep_rollout(self.device)

    def reset_context(self, env_dones=None):
        """
        Reset language contexts.
        :param env_dones (np.ndarray): Done state for each parallel environment,
            default None.
        """
        if self.lang_contexts is None:
            self.lang_contexts = np.zeros(
                (self.n_envs, self.context_dim), dtype=np.float32)
        else:
            assert env_dones is not None
            self.lang_contexts = self.lang_contexts * (1 - env_dones).astype(
                np.float32)[..., np.newaxis]

    def reset_policy_buffer(self):
        self.comm_n_act_policy.reset_buffer()

    def store_language_inputs(self, obs, parsed_obs):
        obs = obs.reshape(-1, obs.shape[-1])
        parsed_obs = [
            sent for env_sent in parsed_obs for sent in env_sent]
        self.lang_learner.store(obs, parsed_obs)

    def store_exp(self, rewards, dones):
        self.comm_n_act_policy.store_act(
            rewards[..., np.newaxis], 
            dones, 
            self.values, 
            self.actions, 
            self.action_log_probs, 
            self.comm_actions, 
            self.comm_action_log_probs, 
            self.rnn_states, 
            self.rnn_states_critic)

    def _make_obs(self, obs):
        """
        Generate observations and shared_observations, with the message 
        contexts concatenated.
        :param obs: (np.ndarray) Local observations, dim=(n_envs, 
            n_agents, obs_dim).
        """
        n_envs = obs.shape[0]
        lang_contexts = self.lang_contexts.reshape(
            n_envs, 1, self.context_dim).repeat(
                self.n_agents, axis=1)

        # Make all possible shared observations
        shared_obs = []
        ids = list(range(self.n_agents)) * 2
        for a_i in range(self.n_agents):
            shared_obs.append(
                obs[:, ids[a_i:a_i + self.n_agents]].reshape(
                    n_envs, 1, -1))

        if self.comm_type == "no_comm":
            shared_obs = np.concatenate(shared_obs, axis=1)            
        else:
            shared_obs = np.concatenate(
                (np.concatenate(shared_obs, axis=1), lang_contexts), axis=-1)
            obs = np.concatenate((obs, lang_contexts), axis=-1)
        
        return obs, shared_obs

    @torch.no_grad()
    def comm_n_act(self, obs, perfect_messages=None):
        """
        Perform a whole model step, with first a round of communication and 
        then choosing action for each agent.

        :param obs (np.ndarray): Observations for each agent in each parallel 
            environment, dim=(n_envs, n_agents, obs_dim).
        :param perfect_messages (list(list(list(str)))): "Perfect" messages 
            given by the parser, default None.

        :return actions (np.ndarray): Actions for each agent, 
            dim=(n_envs, n_agents, 1).
        :return broadcasts (list(list(str))): List of broadcasted messages for
            each parallel environment.
        :return agent_messages: (list(list(str))): Messages generated by each 
            agent.
        """
        obs, shared_obs = self._make_obs(obs)
        self.comm_n_act_policy.store_obs(obs, shared_obs)

        # Get actions
        self.values, self.actions, self.action_log_probs, self.comm_actions, \
            self.comm_action_log_probs, self.rnn_states, \
            self.rnn_states_critic = self.comm_n_act_policy.get_actions()

        # Get messages
        if self.comm_type in ["language", "emergent_discrete"]:
            messages = self.lang_learner.generate_sentences(
                np.concatenate(self.comm_actions))

            # Arrange messages by env and construct broadcasts
            broadcasts = []
            messages_by_env = []
            for e_i in range(self.n_envs):
                env_broadcast = []
                for a_i in range(self.n_agents):
                    env_broadcast.extend(messages[e_i * self.n_agents + a_i])
                broadcasts.append(env_broadcast)
                messages_by_env.append(messages[
                    e_i * self.n_agents:e_i * self.n_agents + self.n_agents])

            # Get lang contexts
            self.lang_contexts = self.lang_learner.encode_sentences(
                broadcasts).cpu().numpy()

        elif self.comm_type == "emergent_continuous":
            messages_by_env = self.comm_actions
            # Get lang contexts
            if self.comm_ec_strategy == "sum":
                self.lang_contexts = self.comm_actions.sum(axis=1)
                broadcasts = self.lang_contexts
            elif self.comm_ec_strategy == "mean":
                self.lang_contexts = self.comm_actions.mean(axis=1)
                broadcasts = self.lang_contexts
            else:
                raise NotImplementedError("Emergent communication strategy not implemented:", self.comm_ec_strategy)

        elif self.comm_type == "perfect_comm":
            assert perfect_messages is not None
            messages_by_env = perfect_messages
            broadcasts = []
            for env_messages in messages_by_env:
                env_broadcast = []
                for message in env_messages:
                    env_broadcast.extend(message)
                broadcasts.append(env_broadcast)
            # Get lang contexts
            self.lang_contexts = self.lang_learner.encode_sentences(
                broadcasts).cpu().numpy()

        elif self.comm_type == "no_comm":
            messages_by_env = None
            broadcasts = None
        else:
            raise NotImplementedError("Communication type not implemented:", self.comm_type)

        # Log communication
        if self.comm_logger is not None:
            self.comm_logger.store_messages(
                obs, 
                messages_by_env, 
                perfect_messages, 
                broadcasts)

        return self.actions, broadcasts, messages_by_env

    def train(self, 
            step, train_policy=True, comm_head_learns_rl=True, train_lang=True):
        self.prep_training()

        warmup = step < self.n_warmup_steps

        losses = {}

        if train_policy:
            losses.update(
                self.comm_n_act_policy.train(warmup, comm_head_learns_rl))
        
        # if self.comm_pol_algo != "no_comm":
        #     comm_pol_losses = self.comm_policy.train(warmup)
        #     for k, l in comm_pol_losses.items():
        #         losses["comm_" + k] = l
        
        if self.comm_type in ["perfect_comm", "language"] and train_lang:
            lang_losses = self.lang_learner.train()
            for k, l in lang_losses.items():
                losses["lang_" + k] = l
        
        return losses

        