import copy
import torch
import numpy as np

from torch import nn

from .modules.lang_learner import LanguageLearner
from .modules.comm_policy import PerfectComm, CommPPO_MLP
from .policy.mappo_contextinobs.mappo import MAPPO
from .policy.mappo_contextinobs.utils import get_shape_from_obs_space


class LMC:
    """
    Language-Memory for Communication using a pre-defined discrete language.
    """
    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 vocab, device):
        self.args = args
        self.n_agents = n_agents
        self.context_dim = args.context_dim
        self.n_parallel_envs = args.n_parallel_envs
        self.n_warmup_steps = args.n_warmup_steps
        self.token_penalty = args.comm_token_penalty
        self.klpretrain_coef = args.comm_klpretrain_coef
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

        self.comm_pol_algo = args.comm_policy_algo
        if self.comm_pol_algo == "ppo_mlp":
            self.comm_policy = CommPPO_MLP(args, n_agents, self.lang_learner)
        elif self.comm_pol_algo == "perfect_comm":
            self.comm_policy = PerfectComm(self.lang_learner)
        elif self.comm_pol_algo == "no_comm":
            self.comm_policy = None
            self.context_dim = 0
        else:
            raise NotImplementedError("Bad name given for communication policy algo.")

        if args.policy_algo == "mappo":
            obs_dim = get_shape_from_obs_space(obs_space[0])
            shared_obs_dim = get_shape_from_obs_space(shared_obs_space[0])
            self.policy = MAPPO(
                args, n_agents, obs_dim + self.context_dim, 
                shared_obs_dim + self.context_dim,
                act_space[0], device)

        self.last_messages = None
        self.last_klpretrain_rewards = None

    def prep_training(self):
        self.lang_learner.prep_training()
        self.policy.prep_training()

    def prep_rollout(self, device=None):
        self.lang_learner.prep_rollout(device)
        self.policy.prep_rollout(device)

    def _make_obs(self, obs, message_contexts):
        n_parallel_envs = obs.shape[0]

        shared_obs = np.concatenate(
            (obs.reshape(n_parallel_envs, -1), message_contexts), 
            axis=-1)
        obs = np.concatenate(
            (obs, message_contexts.reshape(
                n_parallel_envs, 1, self.context_dim).repeat(
                    self.n_agents, axis=1)), 
            axis=-1)
        return obs, shared_obs

    def start_episode(self):
        self.policy.start_episode()

    def comm_n_act(self, obs, lang_contexts, perfect_messages=None):
        # Get messages
        if self.comm_policy is not None:
            broadcasts, messages, lang_contexts, klpretrain_rewards = \
                self.comm_policy.comm_step(obs, lang_contexts, perfect_messages)
        else:
            lang_contexts = np.zeros((self.n_parallel_envs, 0))
            broadcasts = []

        # Save messages and klpretrain_rewards for communication evaluation
        self.last_messages = messages
        self.last_klpretrain_rewards = klpretrain_rewards

        # Store policy inputs in policy buffer
        obs, shared_obs = self._make_obs(obs, lang_contexts)
        self.policy.store_obs(obs, shared_obs)

        # Get actions
        values, actions, action_log_probs, rnn_states, rnn_states_critic = \
            self.policy.get_actions()

        return values, actions, action_log_probs, rnn_states, \
               rnn_states_critic, broadcasts, lang_contexts

    def eval_comm(self, env_rewards):
        if self.comm_policy is not None:
            print("MESSAGES", self.last_messages, len(self.last_messages))
            print("Env rewards", env_rewards, env_rewards.shape)
            print("klpretrain", self.last_klpretrain_rewards, self.last_klpretrain_rewards.shape, type(self.last_klpretrain_rewards))
            token_penalties = np.ones_like(
                self.last_klpretrain_rewards) * -self.token_penalty
            print("PENALTIES", token_penalties)
            token_rewards = self.klpretrain_coef * self.last_klpretrain_rewards \
                             + token_penalties
            print("REWARDS", token_rewards, token_rewards.shape)
            self.comm_policy.store_rewards(env_rewards.flatten(), token_rewards)
            #self.comm_policy.store_rewards()

    def reset_context(self, current_lang_contexts=None, env_dones=None):
        """
        Returns reset language contexts.
        :param current_lang_contexts (np.ndarray): default None, if not 
            provided return zero-filled contexts, if provided return contexts
            with zeros where the env is done.
        :param env_dones (np.ndarray): Done state for each parallel environment,
            default None, must be provided if current_lang_contexts is.

        :return lang_contexts (np.ndaray): new language contexts.
        """
        if current_lang_contexts is None:
            return np.zeros(
                (self.n_parallel_envs, self.context_dim), dtype=np.float32)
        else:
            assert env_dones is not None, "env_dones must be provided if current_lang_contexts is."
            return current_lang_contexts * (1 - env_dones).astype(
                np.float32)[..., np.newaxis]

    def store_exp(self, rewards, dones, infos, values, 
            actions, action_log_probs, rnn_states, rnn_states_critic):
        self.policy.store_act(
            rewards, dones, infos, values, actions, action_log_probs, 
            rnn_states, rnn_states_critic)

    def store_language_inputs(self, obs, parsed_obs):
        obs = obs.reshape(-1, obs.shape[-1])
        parsed_obs = [
            sent for env_sent in parsed_obs for sent in env_sent 
            if len(sent) > 0]
        self.lang_learner.store(obs, parsed_obs)

    def train(self, step):
        self.prep_training()
        # Train policy
        warmup = step < self.n_warmup_steps
        pol_losses = self.policy.train(warmup)
        # Train language
        if self.comm_policy is not None:
            lang_losses = self.lang_learner.train()
            return pol_losses, lang_losses
        else:
            return pol_losses

    def save(self, path):
        save_dict = self.policy.get_save_dict()
        save_dict.update(self.lang_learner.get_save_dict())
        if self.comm_policy is not None:
            save_dict.update(self.comm_policy.get_save_dict())
        torch.save(save_dict, path)
