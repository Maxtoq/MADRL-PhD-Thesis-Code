import torch
import numpy as np

from .language.lang_learner import LanguageLearner
from .policy.acc_mappo import ACC_MAPPO
from .policy.acc_buffer import ACC_ReplayBuffer
from .policy.acc_trainer import ACC_Trainer
from .policy.utils import get_shape_from_obs_space, torch2numpy, update_linear_schedule


class LanguageGroundedMARL:

    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 parser, device="cpu", comm_logger=None):
        self.n_agents = n_agents
        self.n_steps = args.n_steps
        self.context_dim = args.context_dim
        self.n_envs = args.n_parallel_envs
        self.n_warmup_steps = args.n_warmup_steps
        self.token_penalty = args.comm_token_penalty
        self.env_reward_coef = args.comm_env_reward_coef
        self.comm_type = args.comm_type
        self.comm_ec_strategy = args.comm_ec_strategy
        self.enc_obs = args.enc_obs
        self.recurrent_N = args.policy_recurrent_N
        self.hidden_dim = args.hidden_dim
        self.comm_logger = comm_logger
        self.device = device

        # Parameters for annealing learning rates
        self.lang_lr = args.lang_lr
        self.lang_lr_anneal_to = args.lang_lr_anneal_to
        self.anneal_lang_lr = self.lang_lr_anneal_to < self.lang_lr \
            and self.comm_type in ["language", "perfect_comm"]

        if self.comm_type == "no_comm":
            policy_input_dim = get_shape_from_obs_space(obs_space[0])
            critic_input_dim = get_shape_from_obs_space(shared_obs_space[0])
        elif self.enc_obs:
            policy_input_dim = self.context_dim * 2
            critic_input_dim = self.context_dim * (self.n_agents + 1)
        else:
            policy_input_dim = \
                get_shape_from_obs_space(obs_space[0]) + self.context_dim
            critic_input_dim = get_shape_from_obs_space(shared_obs_space[0]) \
                                + self.context_dim
        act_dim = act_space[0].n

        # Modules
        obs_dim = obs_space[0].shape[0]
        self.lang_learner = LanguageLearner(
            args,
            obs_dim, 
            self.context_dim,
            parser, 
            device)

        self.acc = ACC_MAPPO(
            args, 
            self.lang_learner, 
            n_agents, 
            policy_input_dim, 
            critic_input_dim, 
            act_dim, 
            device)

        self.buffer = ACC_ReplayBuffer(
            args, 
            n_agents, 
            policy_input_dim, 
            critic_input_dim,
            1, 
            self.context_dim, 
            obs_dim)

        self.trainer = ACC_Trainer(
            args, self.acc.agents, self.lang_learner, self.buffer, self.device)

        # Language context, to carry to next steps
        self.lang_contexts = np.zeros(
            (self.n_envs, self.context_dim), dtype=np.float32)
        # Messages rewards
        self.comm_rewards = None
        # Matrices used during rollout
        self.act_values = None
        self.comm_values = None
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
        self.acc.prep_training(self.device)
        self.trainer.device = self.device
        if self.trainer.act_value_normalizer is not None:
            self.trainer.act_value_normalizer.to(self.device)
            self.trainer.comm_value_normalizer.to(self.device)

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        self.lang_learner.prep_rollout(self.device)
        self.acc.prep_rollout(self.device)
        self.trainer.device = self.device
        if self.trainer.act_value_normalizer is not None:
            self.trainer.act_value_normalizer.to(self.device)
            self.trainer.comm_value_normalizer.to(self.device)

    def reset_context(self, env_dones):
        """
        Reset language contexts.
        :param env_dones (np.ndarray): Done state for each parallel environment
        """
        self.lang_contexts = self.lang_contexts * (1 - env_dones).astype(
            np.float32)[..., np.newaxis]

    def _make_acc_inputs(self, obs):
        """
        Generate inputs for the ACC agents.
        :param obs: (np.ndarray) Local observations, dim=(n_envs, 
            n_agents, obs_dim).
        """
        lang_contexts = self.lang_contexts.reshape(
            self.n_envs, 1, self.context_dim).repeat(
                self.n_agents, axis=1)

        # Make all possible shared observations
        # critic_input = []
        # ids = list(range(self.n_agents)) * 2
        # for a_i in range(self.n_agents):
        #     critic_input.append(
        #         obs[:, ids[a_i:a_i + self.n_agents]].reshape(
        #             n_envs, 1, -1))

        if self.enc_obs and self.comm_type in ["perfect_comm", "language"]:
            obs = torch.from_numpy(obs).reshape(
                    self.n_envs * self.n_agents, -1).to(
                        self.device, dtype=torch.float32)
            policy_input = self.lang_learner.encode_observations(obs)
            policy_input = torch2numpy(
                policy_input.reshape(self.n_envs, self.n_agents, -1))
        else:
            policy_input = obs.copy()
        
        critic_input = policy_input.reshape(self.n_envs, -1).repeat(
            4, 0).reshape(self.n_envs, self.n_agents, -1)
        
            # critic_input = np.concatenate(critic_input, axis=1)
        if self.comm_type != "no_comm":
            critic_input = np.concatenate(
                (critic_input, lang_contexts), axis=-1)
            policy_input = np.concatenate(
                (policy_input, lang_contexts), axis=-1)
        
        return policy_input, critic_input

    def _store_obs(self, obs, parsed_obs):
        """
        Store observations in replay buffer.
        :param obs: (np.ndarray) Observations for each agent, 
            dim=(n_envs, n_agents, obs_dim).
        :param parsed_obs: (list(list(list(str)))) Sentences parsed from 
            observations, dim=(n_envs, n_agents, len(sentence)).
        """
        policy_input, critic_input = self._make_acc_inputs(obs)
        self.buffer.insert_obs(policy_input, critic_input, obs, parsed_obs)

    def init_episode(self, obs=None, parsed_obs=None):
        # If obs is given -> very first step of all training
        if obs is not None:
            self.buffer.reset_episode()
            self._store_obs(obs, parsed_obs)
        # Else -> reset after rollout, we start with the last step of previous 
        # rollout
        else:
            self.buffer.start_new_episode()

    def store_exp(self, next_obs, next_parsed_obs, act_rewards, dones):
        # Reset rnn_states and masks for done environments
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
        
        # Insert action data in buffer
        self.buffer.insert_act(
            self.rnn_states,
            self.rnn_states_critic,
            self.actions,
            self.action_log_probs,
            self.comm_actions,
            self.comm_action_log_probs,
            self.act_values,
            self.comm_values,
            act_rewards[..., np.newaxis],
            self.comm_rewards,
            masks)

        # Insert next obs in buffer
        self._store_obs(next_obs, next_parsed_obs)

    def _reward_comm(self, messages):
        if self.comm_type == "emergent-continuous":
            return self.comm_rewards

        # Penalty for message length
        for e_i in range(self.n_envs):
            for a_i in range(self.n_agents):
                self.comm_rewards[e_i, a_i] = \
                    -len(messages[e_i][a_i]) * self.token_penalty
        comm_rewards = {
            "message_len": self.comm_rewards.mean()}

        return comm_rewards

    @torch.no_grad()
    def comm_n_act(self, perfect_messages=None, gen_messages=True):
        """
        Perform a whole model step, with first a round of communication and 
        then choosing action for each agent.

        :param perfect_messages (list(list(list(str)))): "Perfect" messages 
            given by the parser, default None.
        :param gen_messages: (np.ndarray) Array of boolean controlling whether
            to generate messages of use perfect messages, one for each parallel
            environment, used for comm_type=language only.

        :return actions (np.ndarray): Actions for each agent, 
            dim=(n_envs, n_agents, 1).
        :return broadcasts (list(list(str))): List of broadcasted messages for
            each parallel environment.
        :return agent_messages: (list(list(str))): Messages generated by each 
            agent.
        """
        # Get actions
        policy_input, critic_input, rnn_states, critic_rnn_states, masks \
            = self.buffer.get_act_params()

        self.act_values, self.comm_values, self.actions, self.action_log_probs, \
            self.comm_actions, self.comm_action_log_probs, self.rnn_states, \
            self.rnn_states_critic = self.acc.get_actions(
                policy_input, critic_input, rnn_states, critic_rnn_states, masks)

        # Get messages
        if self.comm_type in ["language"]:
            messages = self.lang_learner.generate_sentences(
                np.concatenate(self.comm_actions))

            # Arrange messages by env and construct broadcasts
            broadcasts = []
            messages_by_env = []
            for e_i in range(self.n_envs):
                if gen_messages[e_i]:
                    env_messages = messages[
                        e_i * self.n_agents:(e_i + 1) * self.n_agents]
                else:
                    env_messages = perfect_messages[e_i]

                env_broadcast = []
                for message in env_messages:
                    env_broadcast.extend(message)

                broadcasts.append(env_broadcast)
                messages_by_env.append(env_messages)

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

        # Reward messages
        self.comm_rewards = np.zeros((self.n_envs, self.n_agents, 1))
        if self.comm_type != "no_comm":
            comm_rewards = self._reward_comm(messages_by_env)
        else:
            comm_rewards = {}

        # Log communication
        if self.comm_logger is not None:
            self.comm_logger.store_messages(
                obs, 
                messages_by_env, 
                perfect_messages, 
                broadcasts)

        return self.actions, broadcasts, messages_by_env, comm_rewards

    def _anneal_lr(self, step):
        if self.anneal_lang_lr:
            new_lr = update_linear_schedule(
                step, self.n_steps, self.lang_lr, self.lang_lr_anneal_to)
            self.acc.update_lrs(new_lr)

    @torch.no_grad()
    def _compute_returns(self):
        critic_input = torch.from_numpy(
                self.buffer.critic_input[-1]).to(self.device)
        critic_rnn_states = torch.from_numpy(
            self.buffer.critic_rnn_states[-1]).to(self.device)
        masks = torch.from_numpy(self.buffer.masks[-1]).to(self.device)

        next_act_values, next_comm_values = self.acc.compute_last_value(
            self.buffer.critic_input[-1], 
            self.buffer.critic_rnn_states[-1],
            self.buffer.masks[-1])

        self.buffer.compute_returns(
            next_act_values, 
            next_comm_values, 
            self.trainer.act_value_normalizer,
            self.trainer.comm_value_normalizer)

    def train(self, step, 
            train_act_head=True, 
            envs_train_comm=None, 
            train_value_head=True,
            train_lang=True):
        self.prep_training()

        self._anneal_lr(step)

        warmup = step < self.n_warmup_steps

        if self.comm_type in ["no_comm", "perfect_comm"]:
            comm_head_learns_rl = False
        else:
            comm_head_learns_rl = True
        if self.comm_type not in ["perfect_comm", "language"]:
            train_lang = False

        # Compute last value
        self._compute_returns()

        # Train 
        losses = self.trainer.train(
            warmup, comm_head_learns_rl, train_lang, envs_train_comm)
        
        return losses

    def save(self, path):
        self.prep_rollout("cpu")
        save_dict = {
            "acc": self.acc.get_save_dict(),
            "lang_learner": self.lang_learner.get_save_dict(),
            "act_vnorm": self.trainer.act_value_normalizer.state_dict(),
            "comm_vnorm": self.trainer.comm_value_normalizer.state_dict()
        }
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=torch.device('cpu'))
        self.acc.load_params(save_dict["acc"])
        self.lang_learner.load_params(save_dict["lang_learner"])
        self.trainer.act_value_normalizer.load_state_dict(params["act_vnorm"])
        self.trainer.commvalue_normalizer.load_state_dict(params["comm_vnorm"])

        