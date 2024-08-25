import torch
import numpy as np

from torch import nn

from src.algo.language.lang_learner import LanguageLearner
from .comm_agent import Comm_Agent
from .utils import torch2numpy, update_lr

class Comm_MAPPO_Shared:

    def __init__(self, args, word_encoder, n_agents, obs_dim, 
                 shared_obs_dim, act_dim, device):
        self.args = args
        self.n_envs = args.n_parallel_envs
        self.n_agents = n_agents
        self.device = device
        self.recurrent_N = args.policy_recurrent_N
        self.share_params = args.share_params
        self.comm_type = args.comm_type

        self.agents = Comm_Agent(
            args, word_encoder, n_agents, obs_dim, shared_obs_dim, act_dim, 
            device)

        self.eval = False

    @torch.no_grad()
    def compute_last_value(self, joint_obs, joint_obs_rnn_states, masks):
        next_act_values, next_comm_values = self.agents.get_values(
            joint_obs.reshape(self.n_envs * self.n_agents, -1),
            joint_obs_rnn_states.reshape(
                self.n_envs * self.n_agents, self.recurrent_N, -1),
            masks.reshape(self.n_envs * self.n_agents, -1))

        next_act_values = torch2numpy(
            next_act_values.reshape(self.n_envs, self.n_agents, -1))
        next_comm_values = torch2numpy(
            next_comm_values.reshape(self.n_envs, self.n_agents, -1))

        return next_act_values, next_comm_values

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        self.agents.eval()
        self.agents.to(self.device)
        self.agents.set_device(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        self.agents.train()
        self.agents.to(self.device)
        self.agents.set_device(self.device)

    def comm_n_act(
            self, obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, 
            comm_rnn_states, masks, deterministic=False, eval_actions=None, 
            eval_comm_actions=None):
        batch_size = obs.shape[0]
        rnn_batch_size = obs_rnn_states.shape[0]
        obs = torch.from_numpy(obs).to(self.device).reshape(
            batch_size * self.n_agents, -1)
        joint_obs = torch.from_numpy(joint_obs).to(self.device).reshape(
            batch_size * self.n_agents, -1)
        obs_rnn_states = torch.from_numpy(obs_rnn_states).to(
            self.device).reshape(rnn_batch_size * self.n_agents, self.recurrent_N, -1)
        joint_obs_rnn_states = torch.from_numpy(
            joint_obs_rnn_states).to(self.device).reshape(
                rnn_batch_size * self.n_agents, self.recurrent_N, -1)
        comm_rnn_states = torch.from_numpy(comm_rnn_states).to(
            self.device).reshape(rnn_batch_size * self.n_agents, self.recurrent_N, -1)
        masks = torch.from_numpy(masks).to(self.device).reshape(
            batch_size * self.n_agents, -1)
        if eval_actions is not None:
            eval_actions = torch.from_numpy(eval_actions).to(
                self.device).reshape(batch_size * self.n_agents, -1)
            eval_comm_actions = torch.from_numpy(eval_comm_actions).to(
                self.device).reshape(batch_size * self.n_agents, -1)
            _eval = True
        else:
            _eval = False

        # TODO: handle perfect messages

        # Generate comm
        messages, enc_obs, enc_joint_obs, comm_actions, \
            comm_action_log_probs, comm_values, new_obs_rnn_states, \
            new_joint_obs_rnn_states, eval_comm_action_log_probs, \
            eval_comm_dist_entropy \
            = self.agents.forward_comm(
                obs, 
                joint_obs, 
                obs_rnn_states, 
                joint_obs_rnn_states, 
                masks, 
                deterministic,
                eval_comm_actions if _eval else None)

        # Aggregate messages
        if self.comm_type == "no_comm":
            messages = None
        elif self.comm_type == "emergent_continuous":
            # Concatenate messages to get broadcast and repeat for all agents
            messages = messages.reshape(batch_size, -1).repeat(
                1, self.n_agents).reshape(batch_size * self.n_agents, -1)

        # Generate actions
        actions, action_log_probs, values, new_comm_rnn_states, \
            eval_action_log_probs, eval_dist_entropy \
            = self.agents.forward_act(
                messages, 
                enc_obs,
                enc_joint_obs,
                comm_rnn_states,
                masks,
                deterministic,
                eval_actions if _eval else None)

        if not _eval:
            actions = torch2numpy(actions.reshape(
                batch_size, self.n_agents, -1))
            action_log_probs = torch2numpy(action_log_probs.reshape(
                batch_size, self.n_agents, -1))
            values = torch2numpy(values.reshape(
                batch_size, self.n_agents, -1))
            comm_actions = torch2numpy(comm_actions.reshape(
                batch_size, self.n_agents, -1))
            comm_action_log_probs = torch2numpy(comm_action_log_probs.reshape(
                batch_size, self.n_agents, -1))
            comm_values = torch2numpy(comm_values.reshape(
                batch_size, self.n_agents, -1))
            new_obs_rnn_states = torch2numpy(new_obs_rnn_states.reshape(
                rnn_batch_size, self.n_agents, self.recurrent_N, -1))
            new_joint_obs_rnn_states = torch2numpy(
                new_joint_obs_rnn_states.reshape(
                    rnn_batch_size, self.n_agents, self.recurrent_N, -1))
            new_comm_rnn_states = torch2numpy(new_comm_rnn_states.reshape(
                rnn_batch_size, self.n_agents, self.recurrent_N, -1))
            
            return actions, action_log_probs, values, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, new_comm_rnn_states

        else:
            actions = actions.reshape(batch_size, self.n_agents, -1)
            action_log_probs = action_log_probs.reshape(
                batch_size, self.n_agents, -1)
            values = values.reshape(batch_size, self.n_agents, -1)
            comm_actions = comm_actions.reshape(batch_size, self.n_agents, -1)
            comm_action_log_probs = comm_action_log_probs.reshape(
                batch_size, self.n_agents, -1)
            comm_values = comm_values.reshape(batch_size, self.n_agents, -1)
            new_obs_rnn_states = new_obs_rnn_states.reshape(
                rnn_batch_size, self.n_agents, self.recurrent_N, -1)
            new_joint_obs_rnn_states = new_joint_obs_rnn_states.reshape(
                    rnn_batch_size, self.n_agents, self.recurrent_N, -1)
            new_comm_rnn_states = new_comm_rnn_states.reshape(
                rnn_batch_size, self.n_agents, self.recurrent_N, -1)
            eval_action_log_probs = eval_action_log_probs.reshape(
                batch_size, self.n_agents, -1)

            if self.comm_type != "no_comm":
                eval_comm_action_log_probs = eval_comm_action_log_probs.reshape(
                    batch_size, self.n_agents, -1)
            else:
                eval_comm_action_log_probs = None
                eval_comm_dist_entropy = None

            return actions, action_log_probs, values, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, new_comm_rnn_states, \
                eval_action_log_probs, eval_dist_entropy, \
                eval_comm_action_log_probs, eval_comm_dist_entropy

    def get_save_dict(self):
        self.prep_rollout("cpu")
        save_dict = {"agents": self.agents.state_dict()}
        return save_dict

    def load_params(self, params):
        self.agents.load_state_dict(params["agents"])



class Comm_MAPPO():

    def __init__(self, args, word_encoder, n_agents, obs_dim, 
                 shared_obs_dim, act_dim, device):
        self.args = args
        self.n_agents = n_agents
        self.device = device
        # self.context_dim = args.context_dim
        # self.n_envs = args.n_parallel_envs
        # self.recurrent_N = args.policy_recurrent_N
        # self.hidden_dim = args.hidden_dim
        # self.lr = args.lr
        self.share_params = args.share_params
        self.comm_type = args.comm_type

        if self.share_params:
            self.agents = [
                Comm_Agent(
                    args, n_agents, obs_dim, shared_obs_dim, act_dim, device)]
        else:
            self.agents = [
                Comm_Agent(
                    args, n_agents, obs_dim, shared_obs_dim, act_dim, device)
                for a_i in range(self.n_agents)]

        if self.comm_type in ["perfect", "language"]:
            self.lang_learner = LanguageLearner(
                args, word_encoder, obs_dim, args.context_dim, n_agents, device)

        self.eval = False

    # @torch.no_grad()
    # def compute_last_value(self, joint_obs, joint_obs_rnn_states, masks):
    #     next_act_values = []
    #     next_comm_values = []
    #     for a_i in range(self.n_agents):
    #         next_act_value, next_comm_value = self.agents[a_i].get_values(
    #             joint_obs[:, a_i], joint_obs_rnn_states[:, a_i], masks[:, a_i])
    #         next_act_values.append(next_act_value)
    #         next_comm_values.append(next_comm_value)
            
    #     next_act_values = torch2numpy(torch.stack(next_act_values, dim=1))
    #     next_comm_values = torch2numpy(torch.stack(next_comm_values, dim=1))

    #     return next_act_values, next_comm_values

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        for a in self.agents:
            a.eval()
            a.to(self.device)
            a.set_device(self.device)
        if self.comm_type in ["perfect", "language"]:
            self.lang_learner.eval()
            self.lang_learner.to(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        for a in self.agents:
            a.train()
            a.to(self.device)
            a.set_device(self.device)
        if self.comm_type in ["perfect", "language"]:
            self.lang_learner.train()
            self.lang_learner.to(self.device)

    def comm_n_act(
            self, obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, 
            comm_rnn_states, masks, perfect_messages, perfect_broadcasts, 
            deterministic=False, eval_actions=None, eval_comm_actions=None):
        obs = torch.from_numpy(obs).to(self.device)
        joint_obs = torch.from_numpy(joint_obs).to(self.device)
        obs_rnn_states = torch.from_numpy(obs_rnn_states).to(self.device)
        joint_obs_rnn_states = torch.from_numpy(
            joint_obs_rnn_states).to(self.device)
        comm_rnn_states = torch.from_numpy(comm_rnn_states).to(self.device)
        perfect_messages = torch.from_numpy(perfect_messages).to(self.device)
        # perfect_broadcasts = torch.from_numpy(perfect_broadcasts).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        if eval_actions is not None:
            eval_actions = torch.from_numpy(eval_actions).to(self.device)
            eval_comm_actions = torch.from_numpy(
                eval_comm_actions).to(self.device)
            _eval = True
        else:
            _eval = False

        # Generate comm
        agents_messages = []
        agents_enc_obs = []
        agents_enc_joint_obs = []
        agents_comm_actions = []
        agents_comm_action_log_probs = []
        agents_comm_values = []
        agents_new_obs_rnn_states = []
        agents_new_joint_obs_rnn_states = []
        agents_eval_comm_action_log_probs = []
        agents_eval_comm_dist_entropy = []
        agents_lang_obs_enc = []
        for a_i in range(self.n_agents):
            messages, enc_obs, enc_joint_obs, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, eval_comm_action_log_probs, \
                eval_comm_dist_entropy \
                = self.agents[a_i].forward_comm(
                    obs[:, a_i], 
                    joint_obs[:, a_i], 
                    obs_rnn_states[:, a_i], 
                    joint_obs_rnn_states[:, a_i], 
                    masks[:, a_i], 
                    perfect_messages[:, a_i],
                    deterministic,
                    eval_comm_actions[:, a_i] if _eval else None)

            agents_messages.append(messages)
            agents_enc_obs.append(enc_obs)
            agents_enc_joint_obs.append(enc_joint_obs)
            agents_comm_actions.append(comm_actions)
            agents_comm_action_log_probs.append(comm_action_log_probs)
            agents_comm_values.append(comm_values)
            agents_new_obs_rnn_states.append(new_obs_rnn_states)
            agents_new_joint_obs_rnn_states.append(new_joint_obs_rnn_states)
            if _eval:
                agents_eval_comm_action_log_probs.append(
                    eval_comm_action_log_probs)
                agents_eval_comm_dist_entropy.append(eval_comm_dist_entropy)

        # Aggregate messages
        if self.comm_type == "no_comm":
            out_messages = None
            in_messages = None
        elif self.comm_type == "emergent_continuous":
            # Concatenate messages to get broadcast
            out_messages = torch.stack(agents_messages, dim=1)
            in_messages = torch.concatenate(agents_messages, 1)
        elif self.comm_type == "perfect":
            out_messages = torch.stack(agents_messages, dim=1)
            in_messages = self.lang_learner.encode_sentences(perfect_broadcasts)

        # Generate actions
        agents_actions = []
        agents_action_log_probs = []
        agents_values = []
        agents_new_comm_rnn_states = []
        agents_eval_action_log_probs = []
        agents_eval_dist_entropy = []
        agents_enc_perf_br = []
        for a_i in range(self.n_agents):
            actions, action_log_probs, values, new_comm_rnn_states, \
                eval_action_log_probs, eval_dist_entropy \
                = self.agents[a_i].forward_act(
                    in_messages, 
                    agents_enc_obs[a_i],
                    agents_enc_joint_obs[a_i],
                    comm_rnn_states[:, a_i],
                    masks[:, a_i],
                    deterministic,
                    eval_actions[:, a_i] if _eval else None)

            agents_actions.append(actions)
            agents_action_log_probs.append(action_log_probs)
            agents_values.append(values)
            agents_new_comm_rnn_states.append(new_comm_rnn_states)
            if _eval:
                agents_eval_action_log_probs.append(eval_action_log_probs)
                agents_eval_dist_entropy.append(eval_dist_entropy)

        if not _eval:
            actions = torch2numpy(torch.stack(agents_actions, dim=1))
            action_log_probs = torch2numpy(
                torch.stack(agents_action_log_probs, dim=1))
            values = torch2numpy(torch.stack(agents_values, dim=1))
            comm_actions = torch2numpy(torch.stack(agents_comm_actions, dim=1))
            comm_action_log_probs = torch2numpy(
                torch.stack(agents_comm_action_log_probs, dim=1))
            comm_values = torch2numpy(torch.stack(agents_comm_values, dim=1))
            new_obs_rnn_states = torch2numpy(
                torch.stack(agents_new_obs_rnn_states, dim=1))
            new_joint_obs_rnn_states = torch2numpy(
                torch.stack(agents_new_joint_obs_rnn_states, dim=1))
            new_comm_rnn_states = torch2numpy(
                torch.stack(agents_new_comm_rnn_states, dim=1))
            
            return actions, action_log_probs, values, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, new_comm_rnn_states, out_messages

        else:
            actions = torch.stack(agents_actions, dim=1)
            action_log_probs = torch.stack(agents_action_log_probs, dim=1)
            values = torch.stack(agents_values, dim=1)
            comm_actions = torch.stack(agents_comm_actions, dim=1)
            comm_action_log_probs = torch.stack(
                agents_comm_action_log_probs, dim=1)
            comm_values = torch.stack(agents_comm_values, dim=1)
            new_obs_rnn_states = torch.stack(agents_new_obs_rnn_states, dim=1)
            new_joint_obs_rnn_states = torch.stack(
                agents_new_joint_obs_rnn_states, dim=1)
            new_comm_rnn_states = torch.stack(agents_new_comm_rnn_states, dim=1)
            eval_action_log_probs = torch.stack(
                agents_eval_action_log_probs, dim=1)
            eval_dist_entropy = torch.stack(agents_eval_dist_entropy).unsqueeze(-1)
            if self.comm_type not in ["no_comm", "perfect", "emergent_continuous"]:
                eval_comm_action_log_probs = torch.stack(
                    agents_eval_comm_action_log_probs, dim=1)
                eval_comm_dist_entropy = torch.stack(
                    agents_eval_comm_dist_entropy).unsqueeze(-1)
            else:
                eval_comm_action_log_probs = None
                eval_comm_dist_entropy = None

            if self.comm_type in ["perfect", "language"]:
                # enc_perf_br = torch.stack(agents_enc_perf_br, dim=1)
                lang_obs_enc = self.lang_learner.obs_encoder(
                    torch.stack(agents_enc_joint_obs, dim=1).to(self.device))
            else:
                lang_obs_enc = None

            return actions, action_log_probs, values, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, new_comm_rnn_states, out_messages, \
                eval_action_log_probs, eval_dist_entropy, \
                eval_comm_action_log_probs, eval_comm_dist_entropy, lang_obs_enc

    def get_save_dict(self):
        self.prep_rollout("cpu")
        save_dict = {
            "agents": [a.state_dict() for a in self.agents]}
        if self.comm_type in ["perfect", "language"]:
            save_dict["lang_learner"] = self.lang_learner.state_dict()
        return save_dict

    def load_params(self, params):
        for a, ap in zip(self.agents, params["agents"]):
            a.load_state_dict(ap)
        if self.comm_type in ["perfect", "language"]:
            self.lang_learner.load_state_dict(params["lang_learner"])
