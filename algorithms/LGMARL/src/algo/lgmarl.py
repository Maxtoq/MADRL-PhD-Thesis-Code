import torch
import numpy as np

from .language.lang_learner import LanguageLearner
from .policy.acc_mappo import ACC_MAPPO
from .policy.acc_buffer import ACC_ReplayBuffer
from .policy.acc_trainer import ACC_Trainer
from .policy.utils import get_shape_from_obs_space, torch2numpy, update_linear_schedule
from .nn_modules.mlp import MLPNetwork
from src.utils.decay import ParameterDecay


class LanguageGroundedMARL:

    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 parser, device="cpu", log_dir=None):
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
            1.0, 0.0001, self.n_steps, "sigmoid", args.comm_eps_smooth)      

        self.comm_encoder = None
        buffer_obs_dim = None

        # Get model input dims
        obs_dim = get_shape_from_obs_space(obs_space[0])
        joint_obs_dim = get_shape_from_obs_space(shared_obs_space[0])
        if self.comm_type == "no_comm":
            policy_input_dim = obs_dim
            critic_input_dim = joint_obs_dim
        elif (self.comm_type == "emergent_continuous" 
                and self.comm_ec_strategy == "cat"):
            policy_input_dim = obs_dim + self.n_agents * args.context_dim
            critic_input_dim = joint_obs_dim + self.n_agents * args.context_dim
        elif (self.comm_type == "emergent_continuous" 
                and self.comm_ec_strategy == "nn"):
            self.comm_encoder = MLPNetwork(
                self.n_agents * args.context_dim,
                args.context_dim,
                self.hidden_dim)
            self.comm_encoder.to(self.device)
            policy_input_dim = obs_dim + args.context_dim
            critic_input_dim = joint_obs_dim + args.context_dim
            buffer_obs_dim = obs_dim + self.n_agents * args.context_dim
        else:
            policy_input_dim = obs_dim + args.context_dim
            critic_input_dim = joint_obs_dim + args.context_dim
        act_dim = act_space[0].n

        # Modules
        obs_dim = obs_space[0].shape[0]
        self.lang_learner = LanguageLearner(
            args,
            obs_dim, 
            args.context_dim,
            parser, 
            n_agents,
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
            policy_input_dim if buffer_obs_dim is None else buffer_obs_dim, 
            critic_input_dim,
            1, 
            args.context_dim, 
            self.lang_learner.word_encoder.max_message_len,
            log_dir)

        self.trainer = ACC_Trainer(
            args, 
            self.acc.agents, 
            self.lang_learner, 
            self.buffer, 
            self.comm_encoder,
            self.device)

        # Language context, to carry to next steps
        comm_act_dim = args.context_dim
        if (self.comm_type == "emergent_continuous" 
                and self.comm_ec_strategy in ["cat", "nn"]):
            comm_act_dim *= self.n_agents
        self.lang_contexts = np.zeros(
            (self.n_envs, comm_act_dim), dtype=np.float32)

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
        self.gen_comm = None

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        self.lang_learner.prep_training(self.device)
        self.acc.prep_training(self.device)
        self.trainer.device = self.device
        if self.trainer.act_value_normalizer is not None:
            self.trainer.act_value_normalizer.to(self.device)
            self.trainer.comm_value_normalizer.to(self.device)

        self.comm_encoder.to(self.device)
        self.comm_encoder.train()

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        self.lang_learner.prep_rollout(self.device)
        self.acc.prep_rollout(self.device)
        self.trainer.device = self.device
        if self.trainer.act_value_normalizer is not None:
            self.trainer.act_value_normalizer.to(self.device)
            self.trainer.comm_value_normalizer.to(self.device)

        self.comm_encoder.to(self.device)
        self.comm_encoder.eval()

    def reset_context(self, env_dones):
        """
        Reset language contexts.
        :param env_dones (np.ndarray): Done state for each parallel environment
        """
        self.lang_contexts = self.lang_contexts * (1 - env_dones).astype(
            np.float32)[..., np.newaxis]

    def _make_acc_inputs(self, obs):
        """
        Generate inputs for the ACC agents:
            - Actor-Communicator takes: local observation + social context,
            - Critic takes: joint observation + social context.
        :param obs: (np.ndarray) Local observations, dim=(n_envs, 
            n_agents, obs_dim).
        """
        lang_contexts = self.lang_contexts.reshape(
            self.n_envs, 1, -1).repeat(
                self.n_agents, axis=1)

        # Make all possible shared observations
        # critic_input = []
        # ids = list(range(self.n_agents)) * 2
        # for a_i in range(self.n_agents):
        #     critic_input.append(
        #         obs[:, ids[a_i:a_i + self.n_agents]].reshape(
        #             n_envs, 1, -1))

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

    def _store_obs(self, obs, perf_messages):
        """
        Store observations in replay buffer.
        :param obs: (np.ndarray) Observations for each agent, 
            dim=(n_envs, n_agents, obs_dim).
        :param perf_messages: (list(list(list(str)))) Sentences parsed from 
            observations, dim=(n_envs, n_agents, len(sentence)).
        """
        # Get inputs for the model
        policy_input, critic_input = self._make_acc_inputs(obs)

        # Encode sentences and build broadcast
        enc_perf_mess, enc_perf_br \
            = self.lang_learner.word_encoder.encode_rollout_step(perf_messages)

        # perf_broadcast = []
        # for env_pm in perf_messages:
        #     env_perf_bc = []
        #     for m in env_pm:
        #         env_perf_bc.extend(m)
        #     perf_broadcast.append([env_perf_bc] * self.n_agents)

        self.buffer.insert_obs(
            policy_input, critic_input, enc_perf_mess, enc_perf_br)

    def init_episode(self, obs=None, perf_messages=None):
        # If obs is given -> very first step of all training
        if obs is not None:
            self.buffer.reset()
            self._store_obs(obs, perf_messages)
        # Else -> reset after rollout, we start with the last step of previous 
        # rollout
        else:
            self.buffer.start_new_episode()

    def store_exp(self, next_obs, next_perf_messages, act_rewards, dones):
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
            masks,
            self.comm_rewards,
            self.gen_comm)

        # Insert next obs in buffer
        self._store_obs(next_obs, next_perf_messages)

    def _reward_comm(self, messages):
        if self.comm_type == "emergent_continuous":
            self.comm_rewards = np.zeros((self.n_envs, self.n_agents, 1))
            return {}

        # Penalty for message length
        lengths = (np.concatenate(
            (messages, np.ones((*messages.shape[:-1], 1))), -1) == 1).argmax(-1)
        self.comm_rewards = (
            np.ones((self.n_envs, self.n_agents)) * lengths * -self.token_penalty
        )[..., np.newaxis]

        comm_rewards = {
            "message_len": self.comm_rewards.mean()}

        return comm_rewards

    @torch.no_grad()
    def comm_n_act(self):
        """
        Perform a whole model step, with first a round of communication and 
        then choosing action for each agent.

        :param env_gen_messages: (np.ndarray) Array of boolean controlling 
            whether to generate messages or use perfect messages, one for each 
            parallel environment, used for comm_type=language only.

        :return actions (np.ndarray): Actions for each agent, 
            dim=(n_envs, n_agents, 1).
        :return broadcasts (list(list(str))): List of broadcasted messages for
            each parallel environment.
        :return agent_messages: (list(list(str))): Messages generated by each 
            agent.
        """
        # Get actions
        policy_input, critic_input, rnn_states, critic_rnn_states, masks, \
            perfect_messages, perfect_broadcasts = self.buffer.get_act_params()

        if self.comm_encoder is not None:
            print(policy_input, policy_input.shape)
            exit()

        self.act_values, self.comm_values, self.actions, self.action_log_probs, \
            self.comm_actions, self.comm_action_log_probs, self.rnn_states, \
            self.rnn_states_critic = self.acc.get_actions(
                policy_input, critic_input, rnn_states, critic_rnn_states, masks)

        # Get messages
        if self.comm_type in ["language"]:
            messages_by_env = self.lang_learner.generate_sentences(np.concatenate(
                self.comm_actions)).reshape(self.n_envs, self.n_agents, -1)

            # Decide which comm strategy 
            self.gen_comm = np.random.random(
                (self.n_envs, self.n_agents, 1)) > self.comm_eps.value

            # Build broadcasts
            broadcasts = []
            for e_i in range(self.n_envs):
                env_br = []
                for a_i in range(self.n_agents):
                    # Replace message by perfect message if not gen_comm
                    if self.gen_comm[e_i, a_i, 0]:
                        agent_m = messages_by_env[e_i, a_i]
                    else:
                        agent_m = perfect_messages[e_i, a_i]

                    # De-pad message and add to broadcast
                    end_i = (np.concatenate((agent_m, [1])) == 1).argmax()
                    env_br.extend(agent_m[:end_i])
                env_br.append(1)
                broadcasts.append(env_br)

            # Get lang contexts
            self.lang_contexts = self.lang_learner.encode_sentences(
                broadcasts).cpu().numpy()

        elif self.comm_type == "perfect_comm":
            messages_by_env = perfect_messages
            
            # Get lang contexts
            broadcasts = [
                env_br[0] for env_br in perfect_broadcasts]

            self.lang_contexts = self.lang_learner.encode_sentences(
                broadcasts).cpu().numpy()

        elif self.comm_type == "emergent_continuous":
            messages_by_env = self.comm_actions
            self.gen_comm = np.ones((self.n_envs, self.n_agents, 1))
            # Get lang contexts
            if self.comm_ec_strategy in ["cat", "nn"]:
                self.lang_contexts = self.comm_actions.reshape(self.n_envs, -1)
            elif self.comm_ec_strategy == "sum":
                self.lang_contexts = self.comm_actions.sum(axis=1)
            elif self.comm_ec_strategy == "mean":
                self.lang_contexts = self.comm_actions.mean(axis=1)
            elif self.comm_ec_strategy == "random":
                rand_ids = np.random.randint(self.n_agents, size=self.n_envs)
                self.lang_contexts = self.comm_actions[
                    np.arange(self.n_envs), rand_ids]
            # elif self.comm_ec_strategy == "nn":
            #     self.lang_contexts = self.comm_encoder(
            #         torch.Tensor(self.comm_actions.reshape(self.n_envs, -1)).to(self.device)).cpu().numpy()
            else:
                raise NotImplementedError("Emergent communication strategy not implemented:", self.comm_ec_strategy)
            broadcasts = self.lang_contexts

        elif self.comm_type == "no_comm":
            messages_by_env = None
            broadcasts = None

        else:
            raise NotImplementedError("Communication type not implemented:", self.comm_type)

        # Reward messages
        if self.comm_type != "no_comm":
            comm_rewards = self._reward_comm(messages_by_env)
        else:
            comm_rewards = {}

        return self.actions, broadcasts, messages_by_env, comm_rewards

    # def _anneal_capt_weight(self, step):
    #     if self.anneal_capt_weight:
    #         self.trainer.capt_loss_weight = self.capt_weight_decay.get_explo_rate(step)

    def _update_comm_eps(self, step):
        if self.comm_type == "language":
            self.comm_eps.get_explo_rate(step)

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
            train_value_head=True,
            train_lang=True):
        self.prep_training()

        # self._anneal_capt_weight(step)
        self._update_comm_eps(step)

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
            warmup, comm_head_learns_rl, train_lang)

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
        if self.comm_type in ["perfect_comm", "language"]:
            self.lang_learner.load_params(save_dict["lang_learner"])
        self.trainer.act_value_normalizer.load_state_dict(save_dict["act_vnorm"])
        self.trainer.comm_value_normalizer.load_state_dict(save_dict["comm_vnorm"])