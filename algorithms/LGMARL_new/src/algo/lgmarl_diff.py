import torch
import numpy as np

from .language.lm import OneHotEncoder
from .policy_diff.comm_mappo import Comm_MAPPO, Comm_MAPPO_Shared
from .policy_diff.buffer import ReplayBuffer
from .policy_diff.trainer import Trainer
from .policy_diff.utils import get_shape_from_obs_space, torch2numpy, update_linear_schedule
from src.utils.decay import ParameterDecay


class LanguageGroundedMARL:

    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 parser, device="cpu", log_dir=None, comm_eps_start=1.0):
        self.n_agents = n_agents
        self.n_steps = args.n_steps
        self.n_envs = args.n_parallel_envs
        self.n_warmup_steps = args.n_warmup_steps
        self.token_penalty = args.comm_token_penalty
        self.env_reward_coef = args.comm_env_reward_coef
        self.comm_type = args.comm_type
        self.comm_ec_strategy = args.comm_ec_strategy
        self.recurrent_N = args.policy_recurrent_N
        self.hidden_dim = args.hidden_dim
        self.device = device

        # Communication Epsilon-greedy
        self.comm_eps = ParameterDecay(
            comm_eps_start, 0.0001, self.n_steps, "sigmoid", args.comm_eps_smooth)      

        # Get model input dims
        obs_shape = get_shape_from_obs_space(obs_space[0])
        joint_obs_shape = get_shape_from_obs_space(shared_obs_space[0])
        act_dim = act_space[0].n

        # Language encoder
        self.word_encoder = OneHotEncoder(
            parser.vocab, parser.max_message_len)

        if args.share_params:
            ModelClass = Comm_MAPPO_Shared
        else:
            ModelClass = Comm_MAPPO
        self.model = ModelClass(
            args, self.word_encoder, n_agents, obs_shape, joint_obs_shape, 
            act_dim, self.device)

        self.buffer = ReplayBuffer(
            args, 
            n_agents, 
            obs_shape, 
            joint_obs_shape,
            1, 
            args.context_dim, 
            self.word_encoder.max_message_len,
            log_dir)

        self.trainer = Trainer(
            args, 
            self.model, 
            self.buffer, 
            self.device)

        self.actions = None
        self.action_log_probs = None
        self.values = None
        self.comm_actions = None
        self.comm_action_log_probs = None
        self.comm_values = None
        self.obs_rnn_states = None
        self.joint_obs_rnn_states = None
        self.comm_rnn_states = None
        self.gen_comm = None

    def act(self, deterministic=False):
        obs, joint_obs, obs_enc_rnn_states, joint_obs_enc_rnn_states, \
            comm_enc_rnn_states, masks, perfect_messages, perfect_broadcasts \
            = self.buffer.get_act_params()

        self.actions, self.action_log_probs, self.values, self.comm_actions, \
            self.comm_action_log_probs, self.comm_values, self.obs_rnn_states, \
            self.joint_obs_rnn_states, self.comm_rnn_states, messages \
            = self.model.comm_n_act(
                obs, joint_obs, obs_enc_rnn_states, joint_obs_enc_rnn_states, 
                comm_enc_rnn_states, masks, perfect_messages,
                perfect_broadcasts, deterministic)

        return self.actions, messages, None, {"len": 0} # TODO: broadcasts, messages_by_env, comm_rewards

    def init_episode(self, obs=None, perf_messages=None):
        # If obs is given -> very first step of all training
        if obs is not None:
            self.buffer.reset()
            self._store_obs(obs, perf_messages)
        # Else -> reset after rollout, we start with the last step of previous 
        # rollout
        else:
            self.buffer.start_new_episode()

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        # self.lang_learner.prep_training(self.device)
        self.model.prep_training(self.device)
        self.trainer.device = self.device
        # if self.trainer.env_value_normalizer is not None:
        #     self.trainer.env_value_normalizer.to(self.device)
        #     self.trainer.comm_value_normalizer.to(self.device)

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        # self.lang_learner.prep_rollout(self.device)
        self.model.prep_rollout(self.device)
        self.trainer.device = self.device
        # if self.trainer.env_value_normalizer is not None:
        #     self.trainer.env_value_normalizer.to(self.device)
        #     self.trainer.comm_value_normalizer.to(self.device)

    def reset_context(self, env_dones):
        """
        Reset language contexts.
        :param env_dones (np.ndarray): Done state for each parallel environment
        """
        self.lang_contexts = self.lang_contexts * (1 - env_dones).astype(
            np.float32)[..., np.newaxis]

    def store_exp(self, next_obs, next_perf_messages, act_rewards, dones):
        # Reset rnn_states and masks for done environments
        self.obs_rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        self.joint_obs_rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        self.comm_rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        masks = np.ones((self.n_envs, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        # TODO
        comm_rewards = np.zeros_like(act_rewards)

        # gen_comm all true for emergent-continuous
        # if self.comm_type == "emergent_continuous":
        #     self.gen_comm = np.ones_like(self.comm_values)
        
        # Insert action data in buffer
        self.buffer.insert_act(
            self.obs_rnn_states,
            self.joint_obs_rnn_states,
            self.comm_rnn_states,
            self.actions,
            self.action_log_probs,
            self.comm_actions,
            self.comm_action_log_probs,
            self.values,
            self.comm_values,
            act_rewards[..., np.newaxis],
            masks,
            comm_rewards[..., np.newaxis],
            self.gen_comm)

        # Insert next obs in buffer
        self._store_obs(next_obs, next_perf_messages)

    def train(self, step, train_lang=True):
        self.prep_training()

        # self._anneal_capt_weight(step)
        self._update_comm_eps(step)

        warmup = step < self.n_warmup_steps

        if self.comm_type in ["no_comm", "perfect", "emergent_continuous"]:
            comm_head_learns_rl = False
        else:
            comm_head_learns_rl = True
        if self.comm_type not in ["perfect", "language"]:
            train_lang = False

        # Compute last value
        self._compute_returns()

        # Train 
        losses = self.trainer.train_diff(
            warmup, comm_head_learns_rl, train_lang)

        return losses

    @torch.no_grad()
    def _compute_returns(self):
        # joint_obs = torch.from_numpy(
        #         self.buffer.joint_obs[-1]).to(self.device)
        # joint_obs_enc_rnn_states = torch.from_numpy(
        #     self.buffer.joint_obs_enc_rnn_states[-1]).to(self.device)
        # masks = torch.from_numpy(self.buffer.masks[-1]).to(self.device)

        # next_env_values, next_comm_values = self.model.compute_last_value(
        #     joint_obs, joint_obs_enc_rnn_states, masks)

        obs, joint_obs, obs_enc_rnn_states, joint_obs_enc_rnn_states, \
            comm_enc_rnn_states, masks, perfect_messages, perfect_broadcasts \
            = self.buffer.get_act_params()
        _, _, next_env_values, _, _, next_comm_values, _, _, _, _ \
            = self.model.comm_n_act(
                obs, joint_obs, obs_enc_rnn_states, joint_obs_enc_rnn_states, 
                comm_enc_rnn_states, masks, perfect_messages,
                perfect_broadcasts, deterministic=True)

        self.buffer.compute_returns(
            next_env_values, 
            next_comm_values, 
            self.trainer.env_value_normalizer,
            self.trainer.comm_value_normalizer)

    def _store_obs(self, obs, perf_messages):
        """
        Store observations in replay buffer.
        :param obs: (np.ndarray) Observations for each agent, 
            dim=(n_envs, n_agents, obs_dim).
        :param perf_messages: (list(list(list(str)))) Sentences parsed from 
            observations, dim=(n_envs, n_agents, len(sentence)).
        """
        # Encode sentences and build broadcast
        enc_perf_mess, enc_perf_br \
            = self.word_encoder.encode_rollout_step(perf_messages)

        joint_obs = obs.reshape(self.n_envs, 1, -1).repeat(
            self.n_agents, 1)

        self.buffer.insert_obs(obs, joint_obs, enc_perf_mess, enc_perf_br)

    def _update_comm_eps(self, step):
        if self.comm_type == "language":
            self.comm_eps.get_explo_rate(step)    

    def save(self, path):
        self.prep_rollout("cpu")
        save_dict = {
            "acc": self.model.get_save_dict(),
            "act_vnorm": self.trainer.env_value_normalizer.state_dict(),
            "comm_vnorm": self.trainer.comm_value_normalizer.state_dict()}
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_params(save_dict["acc"])
        self.trainer.env_value_normalizer.load_state_dict(save_dict["act_vnorm"])
        self.trainer.comm_value_normalizer.load_state_dict(save_dict["comm_vnorm"])