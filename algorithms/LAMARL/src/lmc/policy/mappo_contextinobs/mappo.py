import torch
import numpy as np
import torch.nn as nn

from itertools import chain

from .r_actor_critic import R_Actor, R_Critic
from .utils import update_linear_schedule, get_gard_norm, huber_loss, mse_loss, check
from .buffer import SeparatedReplayBuffer
from .nn_modules.valuenorm import ValueNorm

##########################################################################
# Code modified from https://github.com/marlbenchmark/on-policy
##########################################################################

def torch2numpy(x):
    return x.detach().cpu().numpy()


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    TODO: changer les types des obs et act space
    :param obs_dim: (int) observation dimension.
    :param cent_obs_dim: (int) value function input dimension (centralized input for MAPPO, decentralized for IPPO).
    :param action_dim: (int) action dimension.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, 
            args, obs_dim, cent_obs_dim, act_space, 
            device=torch.device("cpu")):
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.warming_up = False

        self.actor = R_Actor(args, obs_dim, act_space, device)
        self.critic = R_Critic(args, cent_obs_dim, device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(
            self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(
            self.critic_optimizer, episode, episodes, self.critic_lr)

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr
            self.warming_up = warmup
            

    def get_actions(self, 
            cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
            available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are
            available to agent (if None, all actions available).
        :param deterministic: (bool) whether the action should be mode of 
            distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, 
            deterministic)

        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, \
            rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for
            critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be 
            reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, 
            cent_obs, obs, rnn_states_actor, rnn_states_critic, 
            action, masks, available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor
        update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for 
            actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for
            critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy
            to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be
            reset.
        :param available_actions: (np.ndarray) denotes which actions are 
            available to agent (if None, all actions available).
        :param active_masks: (torch.Tensor) denotes whether an agent is active 
            or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input
            actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for
            the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, 
            active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, 
            obs, rnn_states_actor, masks, 
            available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for
            actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be
            reset.
        :param available_actions: (np.ndarray) denotes which actions are 
            available to agent (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of 
            distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


class R_MAPPOTrainAlgo():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, 
        policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, 
            values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from
            data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or
            dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + \
            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(
                return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(
                return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / \
                active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        shared_obs_batch, obs_batch, rnn_states_batch, \
            rnn_states_critic_batch, actions_batch, value_preds_batch, \
            return_batch, masks_batch, active_masks_batch, \
            old_action_log_probs_batch, adv_targ, \
            available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            shared_obs_batch, obs_batch, rnn_states_batch, 
            rnn_states_critic_batch, actions_batch, masks_batch, 
            available_actions_batch,active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 
            1.0 - self.clip_param, 
            1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) 
                * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, \
            actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True, warmup=False):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training 
            update (e.g. loss, grad norms, etc).
        """
        self.policy.warmup_lr(warmup)
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, \
                    actor_grad_norm, imp_weights = self.ppo_update(
                    sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self, device):
        self.policy.actor.train()
        self.policy.actor.to(device)
        self.policy.actor.tpdv["device"] = device
        self.policy.critic.train()
        self.policy.critic.to(device)
        self.policy.critic.tpdv["device"] = device

    def prep_rollout(self, device):
        self.policy.actor.eval()
        self.policy.actor.to(device)
        self.policy.actor.tpdv["device"] = device
        self.policy.critic.eval()
        self.policy.critic.to(device)
        self.policy.critic.tpdv["device"] = device


class MAPPO:
    """
    Class handling training of MAPPO from paper "The Surprising Effectiveness 
    of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955).
    :param args: (argparse.Namespace) all arguments for training
    :param n_agents: (int) number of agents
    :param obs_dim: (int) observation dimensions, for each agent
    :param cent_obs_dim: (int) centralized observation dimensions, for 
        each agent
    :param act_space: (gym.Space) action dimensions, for each agent
    :param device: (torch.device) cuda device used for training
    """
    def __init__(self, 
            args, n_agents, obs_dim, cent_obs_dim, act_space, 
            device):
        self.args = args
        self.n_agents = n_agents
        self.use_centralized_V = self.args.use_centralized_V
        self.train_device = device

        # Set variant
        if self.args.policy_algo == "rmappo":
            self.args.use_recurrent_policy = True
            self.args.use_naive_recurrent_policy = False
        elif self.args.policy_algo == "mappo":
            self.args.use_recurrent_policy = False 
            self.args.use_naive_recurrent_policy = False
        elif self.args.policy_algo == "ippo":
            self.use_centralized_V = False
        else:
            raise NotImplementedError("Bad param given for policy_algo.")

        # Init agent policies, train algo and buffer
        self.policy = []
        self.trainer = []
        self.buffer = []
        for a_i in range(self.n_agents):
            if self.use_centralized_V:
                shared_obs_dim = cent_obs_dim
            else:
                shared_obs_dim = obs_dim
            # Policy network
            po = R_MAPPOPolicy(
                self.args,
                obs_dim,
                shared_obs_dim,
                act_space,
                device=device)
            self.policy.append(po)

            # Algorithm
            tr = R_MAPPOTrainAlgo(
                self.args, po, device=device)
            self.trainer.append(tr)

            # Buffer
            bu = SeparatedReplayBuffer(
                self.args, 
                obs_dim, 
                shared_obs_dim,
                act_space)
            self.buffer.append(bu)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.train_device
        for tr in self.trainer:
            tr.prep_rollout(device)

    def prep_training(self):
        for tr in self.trainer:
            tr.prep_training(self.train_device)

    def start_episode(self):
        # """
        # Initialize the buffer with first observations.
        # :param obs: (numpy.ndarray) first observations
        # """
        for a_i in range(self.n_agents):
            self.buffer[a_i].reset_episode()
            # if not self.use_centralized_V:
            #     shared_obs = np.array(list(obs[:, a_i]))
            # self.buffer[a_i].shared_obs[0] = shared_obs.copy()
            # self.buffer[a_i].obs[0] = np.array(list(obs[:, a_i])).copy()

    @torch.no_grad()
    def get_actions(self):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for a_i in range(self.n_agents):
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[a_i].policy.get_actions(
                    *self.buffer[a_i].get_act_params())
            # [agents, envs, dim]
            values.append(torch2numpy(value))
            action = torch2numpy(action)            

            actions.append(action)
            action_log_probs.append(torch2numpy(action_log_prob))
            rnn_states.append(torch2numpy(rnn_state))
            rnn_states_critic.append(torch2numpy(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, \
               rnn_states_critic

    def store_obs(self, obs, shared_obs):
        for a_i in range(self.n_agents):
            if not self.use_centralized_V:
                shared_obs = np.array(list(obs[:, a_i]))
            self.buffer[a_i].insert_obs(
                np.array(list(obs[:, a_i])).copy(),
                shared_obs.copy())

    def store_act(self, rewards, dones, infos, values, actions, 
            action_log_probs, rnn_states, rnn_states_critic):
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32)
        masks = np.ones(
            (self.args.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        for a_i in range(self.n_agents):
            # if not self.use_centralized_V:
            #     shared_obs = np.array(list(obs[:, a_i]))
            self.buffer[a_i].insert_act(
                rnn_states[:, a_i],
                rnn_states_critic[:, a_i],
                actions[:, a_i],
                action_log_probs[:, a_i],
                values[:, a_i],
                rewards[:, a_i],
                masks[:, a_i])

    @torch.no_grad()
    def compute_last_value(self):
        for a_i in range(self.n_agents):
            next_value = self.trainer[a_i].policy.get_values(
                self.buffer[a_i].shared_obs[-1],
                self.buffer[a_i].rnn_states_critic[-1],
                self.buffer[a_i].masks[-1])
            next_value = torch2numpy(next_value)
            self.buffer[a_i].compute_returns(
                next_value, self.trainer[a_i].value_normalizer)

    def train(self, warmup=False):
        # Compute last value
        self.compute_last_value()
        # Train
        self.prep_training()
        train_infos = []
        for a_i in range(self.n_agents):
            train_info = self.trainer[a_i].train(
                self.buffer[a_i], warmup=warmup)
            train_infos.append(train_info)
        return train_infos

    def get_save_dict(self):
        self.prep_rollout("cpu")
        agents_params = []
        for a_i in range(self.n_agents):
            params = {
                "actor": self.trainer[a_i].policy.actor.state_dict(),
                "critic": self.trainer[a_i].policy.critic.state_dict()
            }
            if self.trainer[a_i]._use_valuenorm:
                params["vnorm"] = self.trainer[a_i].value_normalizer.state_dict()
            agents_params.append(params)
        save_dict = {
            "agents_params": agents_params
        }
        return save_dict

    def save(self, path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, path)

