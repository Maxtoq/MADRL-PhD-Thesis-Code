import copy
import torch
import random
import numpy as np

from src.algo.language.lang_learner import LanguageLearner
from src.algo.policy_diff.autoencoder import Decoder
from .comm_agent import CommAgent
from .utils import torch2numpy, update_lr
from .langground import LanguageGrounder


class CommMAPPO():

    def __init__(self, args, parser, n_agents, obs_dim, 
                 shared_obs_dim, act_dim, device, block_comm=False, 
                 discrete_action=True):
        self.args = args
        self.n_envs = args.n_parallel_envs
        self.n_agents = n_agents
        self.device = device
        self.block_comm = block_comm
        self.share_params = args.share_params
        self.comm_type = args.comm_type
        self.comm_autoencode = "AE" in self.comm_type
        self.comm_langground = "LG" in self.comm_type

        if self.share_params:
            self.agents = [
                CommAgent(
                    args, parser, n_agents, obs_dim, shared_obs_dim, act_dim, 
                    device, discrete_action)]
        else:
            self.agents = [
                CommAgent(
                    args, parser, n_agents, obs_dim, shared_obs_dim, act_dim, 
                    device, discrete_action)
                for a_i in range(self.n_agents)]

        if self.comm_type == "emergent_discrete_lang":
            vocab = [str(i) for i in range(args.comm_emdisc_max_len)]
            max_message_len = args.comm_emdisc_max_len
            use_gumbel = True
        else:
            vocab = parser.vocab
            max_message_len = parser.max_message_len
            use_gumbel = False
        self.lang_learner = LanguageLearner(
            args, vocab, max_message_len, use_gumbel, device)
        
        if self.comm_autoencode:
            self.obs_decoder = Decoder(args, args.context_dim, obs_dim, device)

        if self.comm_langground:
            assert args.comm_langground_pt is not None, "Need pre-trained language encoder path for LangGround."
            self.lang_ground = LanguageGrounder(
                obs_dim, args.context_dim, args.lang_hidden_dim, args.lang_embed_dim, 
                args.policy_layer_N, args.lang_lr, vocab, max_message_len, device)
            self.lang_ground.load_params(torch.load(args.comm_langground_pt))

        self.eval = False

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        for a in self.agents:
            a.eval()
            a.to(self.device)
            a.set_device(self.device)
        if self.comm_type in ["perfect", "language_sup", "language_rl", 
                "emergent_discrete_lang", "no_comm+lang", "perfect+no_lang",
                "lang+no_clip"]:
            if type(self.lang_learner) == list:
                for ll in self.lang_learner:
                    ll.prep_rollout(self.device)
            else:
                self.lang_learner.prep_rollout(self.device)
        if self.comm_autoencode:
            self.obs_decoder.prep_rollout(self.device)
        if self.comm_langground:
            self.lang_ground.prep_rollout(device)

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        for a in self.agents:
            a.train()
            a.to(self.device)
            a.set_device(self.device)
        if self.comm_type in ["perfect", "language_sup", "language_rl", 
                "emergent_discrete_lang", "no_comm+lang", "perfect+no_lang",
                "lang+no_clip"]:
            if type(self.lang_learner) == list:
                for ll in self.lang_learner:
                    ll.prep_training(self.device)
            else:
                self.lang_learner.prep_training(self.device)
        if self.comm_autoencode:
            self.obs_decoder.prep_training(self.device)
        if self.comm_langground:
            self.lang_ground.prep_training(device)

    def _comm_step(
            self, obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, 
            masks, perfect_messages, deterministic, eval_comm_actions):
        # Generate comm
        all_messages = []
        all_enc_obs = []
        all_enc_joint_obs = []
        all_comm_actions = []
        all_comm_action_log_probs = []
        all_comm_values = []
        all_new_obs_rnn_states = []
        all_new_joint_obs_rnn_states = []
        all_eval_comm_action_log_probs = []
        all_eval_comm_dist_entropy = []
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
                    eval_comm_actions[:, a_i] 
                        if eval_comm_actions is not None else None)

            all_messages.append(messages)
            all_enc_obs.append(enc_obs)
            all_enc_joint_obs.append(enc_joint_obs)
            all_comm_actions.append(comm_actions)
            all_comm_action_log_probs.append(comm_action_log_probs)
            all_comm_values.append(comm_values)
            all_new_obs_rnn_states.append(new_obs_rnn_states)
            all_new_joint_obs_rnn_states.append(new_joint_obs_rnn_states)
            all_eval_comm_action_log_probs.append(eval_comm_action_log_probs)
            all_eval_comm_dist_entropy.append(eval_comm_dist_entropy)
        
        return all_messages, all_enc_obs, all_enc_joint_obs, all_comm_actions, \
            all_comm_action_log_probs, all_comm_values, all_new_obs_rnn_states, \
            all_new_joint_obs_rnn_states, all_eval_comm_action_log_probs, \
            all_eval_comm_dist_entropy

    def _make_broadcasts(self, messages, perfect_messages, gen_comm, lang_input=None):
        broadcasts = []
        for e_i in range(messages.shape[0]):
            env_br = []

            if not self.block_comm:
                for a_i in range(self.n_agents):
                    # Replace message by perfect message if not gen_comm, or if 
                    # gen_comm is not provided (means we do emergent comm)
                    if gen_comm is None or gen_comm[e_i, a_i, 0]:
                        agent_m = messages[e_i, a_i]
                    else:
                        agent_m = perfect_messages[e_i, a_i]

                    # De-pad message and add to broadcast
                    end_i = (np.concatenate((agent_m, [1])) == 1).argmax()
                    env_br.extend(agent_m[:end_i])

            if lang_input is not None:
                env_br.extend(lang_input[e_i])
            else:
                env_br.append(1)
            broadcasts.append(env_br)
        
        return broadcasts

    def _aggreg_messages(
            self, messages, comm_actions, perfect_messages, perfect_broadcasts, 
            comm_eps, eval_gen_comm, lang_input):
        batch_size = len(perfect_messages)

        if lang_input is not None:
            _, lang_input = self.encode_perf_messages(lang_input)

        out_messages = None
        in_messages = np.zeros((batch_size, self.n_agents, 1))
        gen_comm = None
        if self.comm_type in [
                "emergent_continuous", "obs", "emergent_continuous_AE", 
                "emergent_continuous_LG"]:
            # Concatenate messages to get broadcast
            out_messages = torch.stack(messages, dim=1)
            in_messages = torch.concatenate(messages, 1).repeat(
                1, self.n_agents).reshape(batch_size, self.n_agents, -1)

        elif self.comm_type == "emergent_discrete_lang":
            # Generate messages
            if type(self.lang_learner) == list:
                out_messages = []
                for ll, ca in zip(self.lang_learner, comm_actions):
                    out_messages.append(ll.generate_sentences(ca, pad_max=True))
                out_messages = np.stack(out_messages, axis=1) # TODO change this to torch
            else:
                dec_in = torch.stack(comm_actions, dim=1).reshape(
                    batch_size * self.n_agents, -1).to(self.device)
                out_messages = self.lang_learner.generate_sentences(dec_in)
                out_messages = out_messages.reshape(
                    batch_size, self.n_agents, -1)

            broadcasts = self._make_broadcasts(
                out_messages, perfect_messages, gen_comm)

            # Encode messages
            if type(self.lang_learner) == list:
                in_messages = []
                for ll in self.lang_learner:
                    in_messages.append(ll.encode_sentences(broadcasts))
                in_messages = torch.stack(in_messages, dim=1)
            else:
                in_messages = self.lang_learner.encode_sentences(broadcasts)
                in_messages = in_messages.repeat(1, self.n_agents).reshape(
                    batch_size, self.n_agents, -1)

        elif "perfect" in self.comm_type:
            out_messages = np.stack(messages, axis=1)
            if type(self.lang_learner) == list:
                in_messages = []
                for ll in self.lang_learner:
                    in_messages.append(ll.encode_sentences(perfect_broadcasts))
                in_messages = torch.stack(in_messages, dim=1)
            else:
                in_messages = self.lang_learner.encode_sentences(perfect_broadcasts)
                in_messages = in_messages.repeat(1, self.n_agents).reshape(
                    batch_size, self.n_agents, -1)

                # if self.comm_type == "perfect+no_lang":
                #     in_messages = in_messages.detach()

        elif self.comm_type in ["language_sup", "lang+no_clip"]:
            # Generate messages
            if type(self.lang_learner) == list:
                out_messages = []
                for ll, ca in zip(self.lang_learner, comm_actions):
                    out_messages.append(ll.generate_sentences(ca, pad_max=True))
                out_messages = np.stack(out_messages, axis=1)
            else:
                dec_in = torch.stack(comm_actions, dim=1).reshape(
                    batch_size * self.n_agents, -1).to(self.device)
                out_messages = self.lang_learner.generate_sentences(dec_in)
                out_messages = out_messages.reshape(
                    batch_size, self.n_agents, -1)

            # Decide which comm strategy for each message
            if eval_gen_comm is None:
                gen_comm = np.random.random(
                    (batch_size, self.n_agents, 1)) > comm_eps
            else:
                gen_comm = eval_gen_comm

            broadcasts = self._make_broadcasts(
                out_messages, perfect_messages, gen_comm, lang_input)

            # Encode messages
            if type(self.lang_learner) == list:
                in_messages = []
                for ll in self.lang_learner:
                    in_messages.append(ll.encode_sentences(broadcasts))
                in_messages = torch.stack(in_messages, dim=1)
            else:
                in_messages = self.lang_learner.encode_sentences(broadcasts)
                in_messages = in_messages.repeat(1, self.n_agents).reshape(
                    batch_size, self.n_agents, -1)

        return out_messages, in_messages, gen_comm

    def _act_step(
            self, in_messages, enc_obs, enc_joint_obs, comm_rnn_states, masks, 
            deterministic, eval_actions):
        all_actions = []
        all_action_log_probs = []
        all_values = []
        all_new_comm_rnn_states = []
        all_eval_action_log_probs = []
        all_eval_dist_entropy = []
        for a_i in range(self.n_agents):
            actions, action_log_probs, values, new_comm_rnn_states, \
                eval_action_log_probs, eval_dist_entropy \
                = self.agents[a_i].forward_act(
                    in_messages[:, a_i], 
                    enc_obs[a_i],
                    enc_joint_obs[a_i],
                    comm_rnn_states[:, a_i],
                    masks[:, a_i],
                    deterministic,
                    eval_actions[:, a_i] 
                        if eval_actions is not None else None)

            all_actions.append(actions)
            all_action_log_probs.append(action_log_probs)
            all_values.append(values)
            all_new_comm_rnn_states.append(new_comm_rnn_states)
            all_eval_action_log_probs.append(eval_action_log_probs)
            all_eval_dist_entropy.append(eval_dist_entropy)
        
        return all_actions, all_action_log_probs, all_values, \
            all_new_comm_rnn_states, all_eval_action_log_probs, \
            all_eval_dist_entropy
        
    def comm_n_act(
            self, obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, 
            comm_rnn_states, masks, perfect_messages, perfect_broadcasts, 
            deterministic=False, eval_actions=None, eval_comm_actions=None,
            comm_eps=None, eval_gen_comm=None, lang_input=None):
        obs = torch.from_numpy(obs).to(self.device)
        joint_obs = torch.from_numpy(joint_obs).to(self.device)
        obs_rnn_states = torch.from_numpy(obs_rnn_states).to(self.device)
        joint_obs_rnn_states = torch.from_numpy(
            joint_obs_rnn_states).to(self.device)
        comm_rnn_states = torch.from_numpy(comm_rnn_states).to(self.device)
        # perfect_messages = torch.from_numpy(perfect_messages).to(self.device)
        # perfect_broadcasts = torch.from_numpy(perfect_broadcasts).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        if eval_actions is not None:
            eval_actions = torch.from_numpy(eval_actions).to(self.device)
            eval_comm_actions = torch.from_numpy(
                eval_comm_actions).to(self.device)
            _training = True
        else:
            _training = False

        # Generate comm
        messages, enc_obs, enc_joint_obs, comm_actions, comm_action_log_probs, \
            comm_values, new_obs_rnn_states, new_joint_obs_rnn_states, \
            eval_comm_action_log_probs, eval_comm_dist_entropy \
            = self._comm_step(
                obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, masks, 
                perfect_messages, deterministic, eval_comm_actions)

        # Aggregate messages
        out_messages, in_messages, gen_comm = self._aggreg_messages(
            messages, comm_actions, perfect_messages, perfect_broadcasts, 
            comm_eps, eval_gen_comm, lang_input)

        # Generate actions
        actions, action_log_probs, values, new_comm_rnn_states, \
            eval_action_log_probs, eval_dist_entropy \
            = self._act_step(
                in_messages, enc_obs, enc_joint_obs, comm_rnn_states, masks, 
                deterministic, eval_actions)

        # Return data in tensors
        if not _training:
            actions = torch2numpy(torch.stack(actions, dim=1))
            action_log_probs = torch2numpy(
                torch.stack(action_log_probs, dim=1))
            values = torch2numpy(torch.stack(values, dim=1))
            comm_actions = torch2numpy(torch.stack(comm_actions, dim=1))
            comm_action_log_probs = torch2numpy(
                torch.stack(comm_action_log_probs, dim=1))
            comm_values = torch2numpy(torch.stack(comm_values, dim=1))
            new_obs_rnn_states = torch2numpy(
                torch.stack(new_obs_rnn_states, dim=1))
            new_joint_obs_rnn_states = torch2numpy(
                torch.stack(new_joint_obs_rnn_states, dim=1))
            new_comm_rnn_states = torch2numpy(
                torch.stack(new_comm_rnn_states, dim=1))
            if type(out_messages) is torch.Tensor:
                out_messages = torch2numpy(out_messages)
            
            return actions, action_log_probs, values, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, new_comm_rnn_states, out_messages, \
                gen_comm

        else:
            actions = torch.stack(actions, dim=1)
            action_log_probs = torch.stack(action_log_probs, dim=1)
            values = torch.stack(values, dim=1)
            comm_actions = torch.stack(comm_actions, dim=1)
            comm_action_log_probs = torch.stack(
                comm_action_log_probs, dim=1)
            comm_values = torch.stack(comm_values, dim=1)
            new_obs_rnn_states = torch.stack(new_obs_rnn_states, dim=1)
            new_joint_obs_rnn_states = torch.stack(
                new_joint_obs_rnn_states, dim=1)
            new_comm_rnn_states = torch.stack(new_comm_rnn_states, dim=1)
            eval_action_log_probs = torch.stack(
                eval_action_log_probs, dim=1)
            eval_dist_entropy = torch.stack(eval_dist_entropy).unsqueeze(-1)
            if self.comm_type == "language_rl":
                eval_comm_action_log_probs = torch.stack(
                    eval_comm_action_log_probs, dim=1)
                eval_comm_dist_entropy = torch.stack(
                    eval_comm_dist_entropy).unsqueeze(-1)
            else:
                eval_comm_action_log_probs = None
                eval_comm_dist_entropy = None

            if self.comm_type in [
                    "perfect", "language_sup", "no_comm+lang", "perfect+no_lang",
                    "lang+no_clip"]:
                # enc_perf_br = torch.stack(enc_perf_br, dim=1)
                # TODO training distributed ll
                if type(self.lang_learner) == list:
                    raise NotImplementedError()
                lang_obs_enc = self.lang_learner.obs_encoder(
                    torch.stack(enc_joint_obs, dim=1).to(self.device))
            else:
                lang_obs_enc = None

            return actions, action_log_probs, values, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states, new_comm_rnn_states, out_messages, \
                eval_action_log_probs, eval_dist_entropy, \
                eval_comm_action_log_probs, eval_comm_dist_entropy, lang_obs_enc

    def encode_perf_messages(self, perf_messages, pad=True):
        # if type(self.lang_learner) == list:
        we = self.lang_learner.word_encoder \
            if type(self.lang_learner) != list \
            else self.lang_learner[0].word_encoder
        
        return we.encode_rollout_step(perf_messages, pad)

    def get_save_dict(self):
        self.prep_rollout("cpu")
        save_dict = {
            "agents": [a.state_dict() for a in self.agents],
            "lang_learner": self.lang_learner.state_dict() 
                if type(self.lang_learner) != list
                else [ll.state_dict() for ll in self.lang_learner]}
        return save_dict

    def load_params(self, params):
        if type(params) is list:
            self.lang_learner = [
                copy.deepcopy(self.lang_learner) for _ in range(self.n_agents)]

            # agent_ids = random.sample(range(self.n_agents), self.n_agents)
            agent_ids = list(range(self.n_agents))

            n_agents_by_p = self.n_agents // len(params)
            assert len(params) * n_agents_by_p == self.n_agents

            for p_i in range(len(params)):
                for a_i in range(n_agents_by_p):
                    i = p_i * n_agents_by_p + a_i
                    # print(i, agent_ids[i], p_i)
                    self.agents[agent_ids[i]].load_state_dict(
                            params[p_i]["acc"]["agents"][agent_ids[i]])
                    if "lang_learner" in params[p_i]["acc"]: # and self.comm_type not in ["no_comm", "emergent_continuous"]:
                        self.lang_learner[agent_ids[i]].load_state_dict(
                            params[p_i]["acc"]["lang_learner"])
        else:
            for a, ap in zip(self.agents, params["acc"]["agents"]):
                a.load_state_dict(ap)

            if self.comm_type in ["perfect", "language_sup", "language_rl", 
                    "emergent_discrete_lang", "no_comm+lang", "perfect+no_lang",
                    "lang+no_clip"]:
                self.lang_learner.load_state_dict(params["acc"]["lang_learner"])
        
