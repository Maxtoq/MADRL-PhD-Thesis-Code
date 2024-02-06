import torch
import numpy as np

from .language.lang_learner import LanguageLearner
from .policy.acc_mappo import ACC_MAPPO
from .policy.acc_buffer import ACC_ReplayBuffer
from .policy.acc_trainer import ACC_Trainer
from .policy.utils import get_shape_from_obs_space, torch2numpy


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
        self.enc_obs = args.enc_obs
        self.comm_logger = comm_logger
        self.device = device

        if args.comm_type == "no_comm":
            obs_dim = get_shape_from_obs_space(obs_space[0])
            shared_obs_dim = get_shape_from_obs_space(shared_obs_space[0])
        elif self.enc_obs:
            obs_dim = self.context_dim * 2
            shared_obs_dim = self.context_dim * (self.n_agents + 1)
        else:
            obs_dim = get_shape_from_obs_space(obs_space[0]) + self.context_dim
            shared_obs_dim = get_shape_from_obs_space(shared_obs_space[0]) \
                                + self.context_dim
        act_dim = act_space[0].n

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
            obs_dim, 
            shared_obs_dim, 
            act_dim, 
            device)

        self.trainer = ACC_Trainer(args, self.lang_learner, self.device)

        self.buffer = ACC_ReplayBuffer(
            self.args, 
            n_agents, 
            obs_dim, 
            shared_obs_dim, 
            1, 
            self.context_dim)

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

    def reset_buffer(self):
        self.buffer.reset_episode()

    def store_language_inputs(self, obs, parsed_obs):
        obs = obs.reshape(-1, obs.shape[-1])
        parsed_obs = [
            sent for env_sent in parsed_obs for sent in env_sent]
        self.lang_learner.store(obs, parsed_obs)

    def _store_obs(self, obs, shared_obs, parsed_obs):
        """
        Store observations in replay buffer.
        :param obs: (np.ndarray) Observations for each agent, 
            dim=(n_envs, n_agents, obs_dim).
        :param shared_obs: (np.ndarray) Centralised observations, 
            dim=(n_envs, n_agents, shared_obs_dim).
        :param parsed_obs: (list(list(list(str)))) Sentences parsed from 
            observations, dim=(n_envs, n_agents, len(sentence)).
        """
        self.buffer.insert_obs(obs, shared_obs, parsed_obs)

    def store_exp(self, rewards, dones):
        self.rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        self.rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        masks = np.ones((self.n_envs, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert_act(
            self.rnn_states,
            self.rnn_states_critic,
            self.actions,
            self.action_log_probs,
            self.comm_actions,
            self.comm_action_log_probs,
            self.values,
            rewards[..., np.newaxis],
            masks)

    def _make_obs(self, obs):
        """
        Generate observations and shared_observations, with the message 
        contexts concatenated.
        :param obs: (np.ndarray) Local observations, dim=(n_envs, 
            n_agents, obs_dim).
        """
        lang_contexts = self.lang_contexts.reshape(
            self.n_envs, 1, self.context_dim).repeat(
                self.n_agents, axis=1)

        # Make all possible shared observations
        # shared_obs = []
        # ids = list(range(self.n_agents)) * 2
        # for a_i in range(self.n_agents):
        #     shared_obs.append(
        #         obs[:, ids[a_i:a_i + self.n_agents]].reshape(
        #             n_envs, 1, -1))

        if self.enc_obs and self.comm_type in ["perfect_comm", "language"]:
            obs = torch.from_numpy(obs).reshape(
                    self.n_envs * self.n_agents, -1).to(
                        self.device, dtype=torch.float32)
            obs = self.lang_learner.encode_observations(obs)
            obs = torch2numpy(obs.reshape(self.n_envs, self.n_agents, -1))
        
        shared_obs = obs.reshape(self.n_envs, -1).repeat(4, 0).reshape(
            self.n_envs, self.n_agents, -1)
        
            # shared_obs = np.concatenate(shared_obs, axis=1)
        if self.comm_type != "no_comm":
            shared_obs = np.concatenate(
                (shared_obs, lang_contexts), axis=-1)
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
        self._store_obs(obs, shared_obs, perfect_messages)

        # Get actions
        obs, shared_obs, rnn_states, critic_rnn_states, masks \
            = self.buffer.get_act_params()
        self.values, self.actions, self.action_log_probs, self.comm_actions, \
            self.comm_action_log_probs, self.rnn_states, \
            self.rnn_states_critic = self.comm_n_act_policy.get_actions(
                obs, shared_obs, rnn_states, critic_rnn_states, masks)

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
            elif self.comm_ec_strategy == "mean":
                self.lang_contexts = self.comm_actions.mean(axis=1)
            elif self.comm_ec_strategy == "random":
                rand_ids = np.random.randint(self.n_agents, size=self.n_envs)
                self.lang_contexts = self.comm_actions[np.arange(self.n_envs), rand_ids]
            else:
                raise NotImplementedError("Emergent communication strategy not implemented:", self.comm_ec_strategy)
            broadcasts = self.lang_contexts

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

    @torch.no_grad()
    def _compute_returns(self):
        shared_obs = torch.from_numpy(
                self.buffer.shared_obs[-1]).to(self.device)
        critic_rnn_states = torch.from_numpy(
            self.buffer.critic_rnn_states[-1]).to(self.device)
        masks = torch.from_numpy(self.buffer.masks[-1]).to(self.device)

        next_values = self.comm_n_act_policy.compute_last_value(
            self.buffer.shared_obs[-1], 
            self.buffer.critic_rnn_states[-1],
            self.buffer.masks[-1])

        self.buffer.compute_returns(
            next_values, self.trainer.value_normalizer)

    def train(self, step, 
            train_act_head=True, 
            comm_head_learns_rl=True, 
            train_value_head=True,
            train_lang=True):
        self.prep_training()

        warmup = step < self.n_warmup_steps

        losses = {}

        if self.comm_type in ["no_comm", "perfect_comm"]:
            comm_head_learns_rl = False
        # Compute last value
        self._compute_returns()

        # TOOODOOOO

        # Train 
        for a in self.agents:
            a.warmup_lr(warmup)
        losses.update(self.trainer.train(self.buffer, train_comm_head))
        
        if self.comm_type in ["perfect_comm", "language"] and train_lang:
            lang_losses = self.lang_learner.train()
            for k, l in lang_losses.items():
                losses["lang_" + k] = l
        
        return losses

    def save(self, path):
        self.prep_rollout("cpu")
        save_dict = {
            "acc": self.comm_n_act_policy.get_save_dict(),
            "lang_learner": self.lang_learner.get_save_dict()
        }
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=torch.device('cpu'))
        self.comm_n_act_policy.load_params(save_dict["acc"])
        self.lang_learner.load_params(save_dict["lang_learner"])

        