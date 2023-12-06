import copy
import torch
import numpy as np

from torch import nn

from .modules.lang_learner import LanguageLearner
from .modules.comm_policy_context import CommPol_Context
from .modules.comm_policy_perfect import PerfectComm
from .modules.shared_mem import SharedMemory
from .policy.mappo.mappo_shared import MAPPO
from .policy.mappo.utils import get_shape_from_obs_space
from .utils import get_mappo_args


class CommEvaluator:

    def __init__(self, args, lang_learner, device="cpu"):
        self.shared_mem = SharedMemory(args, 10, lang_learner, device)


class LMC:
    """
    Language-Memory for Communication using a pre-defined discrete language.
    """
    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 vocab, global_state_dim=None, device="cpu", comm_logger=None):
        self.args = args
        self.n_agents = n_agents
        self.context_dim = args.context_dim
        self.n_parallel_envs = args.n_parallel_envs
        self.n_warmup_steps = args.n_warmup_steps
        self.comm_n_warmup_steps = args.comm_n_warmup_steps
        self.token_penalty = args.comm_token_penalty
        # self.klpretrain_coef = args.comm_klpretrain_coef
        self.env_reward_coef = args.comm_env_reward_coef
        self.shared_mem_coef = args.comm_shared_mem_coef
        self.shared_mem_reward_type = args.comm_shared_mem_reward_type
        self.obs_dist_coef = args.comm_obs_dist_coef
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

        self.comm_pol_algo = args.comm_policy_algo
        if self.comm_pol_algo == "context_mappo":
            self.comm_policy = CommPol_Context(
                args, self.n_agents, self.lang_learner, device)
        elif self.comm_pol_algo == "perfect_comm":
            self.comm_policy = PerfectComm(self.lang_learner)
        elif self.comm_pol_algo == "no_comm":
            self.comm_policy = None
            self.context_dim = 0
        else:
            raise NotImplementedError("Bad name given for communication policy algo.")

        if global_state_dim is not None:
            self.shared_mem = SharedMemory(args, global_state_dim, device)
            if self.shared_mem_reward_type == "shaping":
                self.last_shm_error = None
        else:
            self.shared_mem = None

        policy_args = get_mappo_args(args)
        if "ppo" in args.policy_algo:
            obs_dim = get_shape_from_obs_space(obs_space[0])
            shared_obs_dim = get_shape_from_obs_space(shared_obs_space[0])
            self.policy = MAPPO(
                policy_args, 
                n_agents, 
                obs_dim + self.context_dim, 
                shared_obs_dim + self.context_dim,
                act_space[0],
                device)

        self.values = None
        self.comm_context = None
        self.action_log_probs = None
        self.rnn_states = None
        self.critic_rnn_states = None
        self.lang_contexts = None

    def prep_training(self):
        self.lang_learner.prep_training()
        self.policy.prep_training()
        if self.comm_policy is not None:
            self.comm_policy.prep_training()
        if self.shared_mem is not None:
            self.shared_mem.prep_training(self.device)

    def prep_rollout(self, device=None):
        self.lang_learner.prep_rollout(device)
        self.policy.prep_rollout(device)
        if self.comm_policy is not None:
            self.comm_policy.prep_rollout(device)
        if self.shared_mem is not None:
            self.shared_mem.prep_rollout(device)

    def _make_obs(self, obs):
        """
        Generate observations and shared_observations, with the message 
        contexts concatenated.
        :param obs: (np.ndarray) Local observations, dim=(n_parallel_envs, 
            n_agents, obs_dim).
        """
        n_parallel_envs = obs.shape[0]
        lang_contexts = self.lang_contexts.reshape(
            n_parallel_envs, 1, self.context_dim).repeat(
                self.n_agents, axis=1)

        # Make all possible shared observations
        shared_obs = []
        ids = list(range(self.n_agents)) * 2
        for a_i in range(self.n_agents):
            shared_obs.append(
                obs[:, ids[a_i:a_i + self.n_agents]].reshape(
                    n_parallel_envs, 1, -1))
        shared_obs = np.concatenate(
            (np.concatenate(shared_obs, axis=1), lang_contexts), axis=-1)

        obs = np.concatenate((obs, lang_contexts), axis=-1)
        
        return obs, shared_obs

    def reset_policy_buffers(self):
        self.policy.reset_buffer()
        self.comm_policy.reset_buffer()

    def comm_n_act(self, obs, perfect_messages=None):
        """
        Perform a whole model step, with first a round of communication and 
        then choosing action for each agent.

        :param obs (np.ndarray): Observations for each agent in each parallel 
            environment, dim=(n_parallel_envs, n_agents, obs_dim).
        :param perfect_messages (list(list(list(str)))): "Perfect" messages 
            given by the parser, default None.

        :return actions (np.ndarray): Actions for each agent, 
            dim=(n_parallel_envs, n_agents, 1).
        :return broadcasts (list(list(str))): List of broadcasted messages for
            each parallel environment.
        :return agent_messages: (list(list(str))): Messages generated by each 
            agent.
        """
        # Get messages
        if self.comm_policy is not None:
            broadcasts, agent_messages, self.lang_contexts = \
                self.comm_policy.comm_step(
                    obs, self.lang_contexts, perfect_messages)
        else:
            self.lang_contexts = np.zeros((self.n_parallel_envs, 0))
            broadcasts = []

        # Log communication
        if self.comm_logger is not None:
            self.comm_logger.store_messages(
                obs, 
                agent_messages, 
                perfect_messages, 
                broadcasts)

        # Store policy inputs in policy buffer
        obs, shared_obs = self._make_obs(obs)
        self.policy.store_obs(obs, shared_obs)

        # Get actions
        self.values, self.actions, self.action_log_probs, self.rnn_states, \
            self.rnn_states_critic = self.policy.get_actions()

        return self.actions, broadcasts, agent_messages

    def _get_shared_mem_reward(self, states):
        """
        :return shared_mem_reward: (np.ndarray) dim=(n_parallel_envs, n_agents)
        """
        shm_error = self.shared_mem.get_prediction_error(
            self.lang_contexts, states)
        if self.shared_mem_reward_type == "direct":
            shared_mem_reward = -np.repeat(
                shm_error[..., np.newaxis], self.n_agents, axis=-1)
        else:
            if self.last_shm_error is None:
                shared_mem_reward = np.zeros(
                    (self.n_parallel_envs, self.n_agents))
            else:
                progress = self.last_shm_error - shm_error
                shared_mem_reward = np.repeat(
                    progress[..., np.newaxis], self.n_agents, axis=-1)
            self.last_shm_error = shm_error
        return shared_mem_reward

    def eval_comm(self, step_rewards, messages, states, dones):
        """
        :param states: (np.ndarray) Global environment states, 
            dim=(n_parallel_envs, state_dim).
        """
        rewards = {"message_reward": step_rewards.mean() * self.env_reward_coef}
        # Log communication rewards
        if self.comm_logger is not None:
            self.comm_logger.store_rewards(step_rewards)

        # Environment reward
        message_rewards = step_rewards * self.env_reward_coef

        # Shared-Memory reward
        if self.shared_mem is not None:
            shm_reward = self._get_shared_mem_reward(states)
            message_rewards += self.shared_mem_coef * shm_reward
            rewards["shm_reward"] = self.shared_mem_coef * shm_reward.mean()

        # Penalty for comm encoding distance to obs encoding
        if self.comm_pol_algo == "context_mappo":
            message_rewards -= self.comm_policy.obs_dist \
                                * self.obs_dist_coef
            rewards["obs_dist"] = self.comm_policy.obs_dist.mean() \
                                    * self.obs_dist_coef

        # Penalty for message length
        message_len = np.array(
            [len(m) for env_m in messages for m in env_m]).reshape(
                message_rewards.shape)
        rewards["message_len"] = message_len.mean() * -self.token_penalty

        tot_rewards = message_rewards * self.env_reward_coef \
                       + message_len * -self.token_penalty
        rewards["tot_rewards"] = tot_rewards.mean()

        self.comm_policy.store_rewards(tot_rewards[..., np.newaxis], dones)

        return rewards

    def train_comm(self, step):
        warmup = step < self.comm_n_warmup_steps
        return self.comm_policy.train(warmup)

    def reset_context(self, env_dones=None):
        """
        Returns reset language contexts.
        :param env_dones (np.ndarray): Done state for each parallel environment,
            default None, must be provided if current_lang_contexts is.
        """
        if self.lang_contexts is None:
            self.lang_contexts = np.zeros(
                (self.n_parallel_envs, self.context_dim), dtype=np.float32)
        else:
            assert env_dones is not None, "env_dones must be provided if current_lang_contexts is."
            self.lang_contexts = self.lang_contexts * (1 - env_dones).astype(
                np.float32)[..., np.newaxis]

        if self.shared_mem is not None:
            self.shared_mem.reset_context(env_dones)

    def store_exp(self, rewards, dones):
        self.policy.store_act(
            rewards[..., np.newaxis], 
            dones, 
            self.values, 
            self.actions, 
            self.action_log_probs, 
            self.rnn_states, 
            self.rnn_states_critic)

    def store_language_inputs(self, obs, parsed_obs):
        obs = obs.reshape(-1, obs.shape[-1])
        parsed_obs = [
            sent for env_sent in parsed_obs for sent in env_sent]
        self.lang_learner.store(obs, parsed_obs)

    def train(self, 
            step, train_policy=True, train_lang=True, train_sharedmem=True):
        self.prep_training()

        warmup = step < self.n_warmup_steps

        losses = {}

        if train_policy:
            losses.update(self.policy.train(warmup))
        
        if self.comm_policy is not None:
            comm_pol_losses = self.comm_policy.train(warmup)
            for k, l in comm_pol_losses.items():
                losses["comm_" + k] = l
        
        if self.comm_policy is not None and train_lang:
            lang_losses = self.lang_learner.train()
            for k, l in lang_losses.items():
                losses["lang_" + k] = l

        if self.shared_mem is not None and train_sharedmem:
            losses["shared_mem"] = self.shared_mem.train()
        
        return losses

    def save(self, path):
        self.prep_rollout("cpu")
        save_dict = self.policy.get_save_dict()
        save_dict.update(self.lang_learner.get_save_dict())
        if self.comm_policy is not None:
            save_dict.update(self.comm_policy.get_save_dict())
        if self.shared_mem is not None:
            save_dict.update(self.shared_mem.get_save_dict())
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=torch.device('cpu'))
        self.policy.load_params(save_dict["agents_params"])
        self.lang_learner.load_params(save_dict)
        if self.comm_pol_algo in ["context_mappo"]:
            self.comm_policy.load_params(save_dict)
        if self.shared_mem is not None:
            self.shared_mem.load_params(save_dict)
