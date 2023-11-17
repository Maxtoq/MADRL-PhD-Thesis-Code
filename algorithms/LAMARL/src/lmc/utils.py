
def torch2numpy(x):
    return x.detach().cpu().numpy()
    
def get_mappo_args(args, args_name=''):
    MAPPO_ARGS = [
        "clip_param",
        "critic_lr",
        "data_chunk_length",
        "entropy_coef",
        "episode_length",
        "gae_lambda",
        "gain",
        "gamma",
        "hidden_size",
        "huber_delta",
        "layer_N",
        "lr",
        "max_grad_norm",
        "n_parallel_envs",
        "num_mini_batch",
        "opti_eps",
        "policy_algo",
        "ppo_epoch",
        "recurrent_N",
        "stacked_frames",
        "use_centralized_V",
        "use_clipped_value_loss",
        "use_feature_normalization",
        "use_gae",
        "use_huber_loss",
        "use_max_grad_norm",
        "use_naive_recurrent_policy",
        "use_orthogonal",
        "use_proper_time_limits",
        "use_policy_active_masks",
        "use_popart",
        "use_recurrent_policy",
        "use_ReLU",
        "use_value_active_masks",
        "use_valuenorm",
        "value_loss_coef",
        "weight_decay"]
    dict_args = {}
    name_found = False
    for a in MAPPO_ARGS:
        if args_name != '' and hasattr(args, args_name + '_' + a):
            dict_args[a] = getattr(args, args_name + '_' + a)
            name_found = True
        elif hasattr(args, a):
            dict_args[a] = getattr(args, a)
        else:
            print("ERROR: argument " + a + " not found.")
            exit()
    
    if args_name != '':
        assert name_found, "Name " + args_name + " not found in arguments."

    assert len(dict_args) == len(MAPPO_ARGS)

    return dict_args