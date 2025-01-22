import os
import json
import time
import torch
import random
import numpy as np


from PIL import Image


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_cuda_device(cfg):
    if torch.cuda.is_available():
        if cfg.cuda_device is None:
            device = torch.device('cuda')
        else:
            device = torch.device(cfg.cuda_device)
        if cfg.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    return device

def load_args(cfg, eval=False):
    if ',' in cfg.model_dir:
        args_path = os.path.join(cfg.model_dir.split(',')[0], "args.txt")
    else:
        args_path = os.path.join(cfg.model_dir, "args.txt")

    if not os.path.isfile(args_path):
        print(f"ERROR: args file {args_path} does not exist.")
        exit()

    else:
        with open(args_path, "r") as f:
            [next(f) for i in range(3)]
            args = json.load(f)

        # For adaptation, modify some parameters
        if cfg.adapt_run or eval:
            # Remove params we want to change from old config
            args.pop("seed")
            args.pop("cuda_device")
            args.pop("model_dir")
            args.pop("use_render")
            args.pop("render_wait_input")
            args.pop("n_steps")
            if "log_comm" in args:
                args.pop("log_comm")
            args.pop("experiment_name")
            args.pop("lr")
            args.pop("lang_lr")
            if "comm_langground_pt" in args:
                args.pop("comm_langground_pt")
            if eval:
                args.pop("n_parallel_envs")
                if "eval_scenario" in args:
                    args.pop("eval_scenario")
                if "n_eval_runs" in args:
                    args.pop("n_eval_runs")
                args.pop("rollout_length")
                args.pop("episode_length")
                # args.pop("n_eval_runs")

            # if "no_render" in args:
            #     args.pop("no_render")

            # Set finetuning parameters
            if cfg.FT_env_name is not None:
                args["env_name"] = cfg.FT_env_name
            if cfg.FT_magym_env_size is not None:
                args["magym_env_size"] = cfg.FT_magym_env_size
            if cfg.FT_magym_not_see_agents is not None:
                args["magym_see_agents"] = not cfg.FT_magym_not_see_agents

            # Load old params in new config
            for a in args:
                if not hasattr(cfg, a):
                    # print(f"WARNING: Argument {a} not found in config.")
                    pass
                else:
                    setattr(cfg, a, args[a])

        # For continuation of existing run, just load all previous parameters 
        # and change number of steps
        elif cfg.continue_run:
            args.pop("model_dir")
            steps_done = args.pop("n_steps")
            args.pop("cuda_device")
            if "continue_run" in args:
                args.pop("continue_run")
            for a in args:
                if not hasattr(cfg, a):
                    print(f"WARNING: Argument {a} not found in config.")
                else:
                    setattr(cfg, a, args[a])
            return steps_done

def render(cfg, envs):
    render = None
    if cfg.save_render:
        render = envs.render("rgb_array")
    if cfg.use_render:
        envs.render("human")

    if cfg.render_wait_input:
        input()
    else:
        time.sleep(0.1)

    return render

def save_frames(frames, dir):
    n_files = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
    path = os.path.join(dir, f"eval_ep{n_files}.gif")
    frames = [Image.fromarray(img.squeeze()) for img in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)

def interact(cfg):
    if cfg.interact:
        add_mess = input("Input message:")
        if len(add_mess):
            add_mess = add_mess.split(" ")
        else:
            add_mess = []
    else:
        add_mess = None
    return add_mess

def log_comm(comm_tab, lang_learner, gen_mess, perf_mess):
    dec_mess = lang_learner.word_encoder.decode_batch(
        gen_mess.reshape(gen_mess.shape[0] * gen_mess.shape[1], -1))
    for e_i in range(gen_mess.shape[0]):
        for a_i in range(gen_mess.shape[1]):
            # print(dec_mess[e_i * gen_mess.shape[1] + a_i], perf_mess[e_i][a_i])
            comm_tab.append({
                "Generated_Message": dec_mess[e_i * gen_mess.shape[1] + a_i], 
                "Perfect_Message": perf_mess[e_i][a_i]})

def count_params(module):
    return sum(p.numel() for p in module.parameters())