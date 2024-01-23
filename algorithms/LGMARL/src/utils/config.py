import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='lmc', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--policy_algo", type=str,
                        default='mappo', choices=["mappo", "rmappo", "ippo"])

    parser.add_argument("--experiment_name", type=str, default="TEST", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda_device", default=None, type=str)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, 
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_parallel_envs", type=int, default=1,
                        help="Number of parallel envs for training rollouts")
    # parser.add_argument("--n_eval_threads", type=int, default=1,
    #                     help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_steps", type=int, default=int(10e6),
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--n_steps_per_update", type=int, default=100,
                        help='Number of environment steps between training updates (default: 100)')

    # env parameters
    parser.add_argument("--env_name", type=str, default='magym_PredPrey', help="specify the names of environment and the task")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=100, help="Max length for any episode")

    # network parameters
    # parser.add_argument("--use_centralized_V", action='store_false',
    #                     default=True, help="Whether to use centralized V function")
    # parser.add_argument("--stacked_frames", type=int, default=1,
    #                     help="Dimension of hidden layers for actor/critic networks")
    # parser.add_argument("--use_stacked_frames", action='store_true',
    #                     default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--policy_layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    # parser.add_argument("--use_ReLU", action='store_false',
    #                     default=True, help="Whether to use ReLU")
    # parser.add_argument("--use_popart", action='store_true', default=False, 
    #                     help="by default False, use PopArt to normalize rewards.")
    # parser.add_argument("--use_valuenorm", action='store_false', default=True, 
    #                     help="by default True, use running mean and std to normalize rewards.")
    # parser.add_argument("--use_feature_normalization", action='store_false',
    #                     default=True, help="Whether to apply layernorm to the inputs")
    # parser.add_argument("--use_orthogonal", action='store_false', default=True,
    #                     help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    # parser.add_argument("--gain", type=float, default=0.01,
    #                     help="The gain # of last action layer")

    # recurrent parameters
    # parser.add_argument("--use_naive_recurrent_policy", action='store_true',
    #                     default=False, help='Whether to use a naive recurrent policy')
    # parser.add_argument("--use_recurrent_policy", action='store_false',
    #                     default=True, help='use a recurrent policy')
    parser.add_argument("--policy_recurrent_N", type=int, default=1, 
                        help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    # parser.add_argument("--critic_lr", type=float, default=5e-4,
    #                     help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--n_warmup_steps", type=int, default=10000)
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    # parser.add_argument("--use_clipped_value_loss",
    #                     action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=2,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    # parser.add_argument("--use_max_grad_norm", action='store_false', default=True, 
    #                     help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    # parser.add_argument("--use_gae", action='store_false',
    #                     default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    # parser.add_argument("--use_proper_time_limits", action='store_true',
    #                     default=False, help='compute returns taking into account time limits')
    # parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    # parser.add_argument("--use_value_active_masks",
    #                     action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    # parser.add_argument("--use_policy_active_masks",
                        # action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=20000, help="number of steps between models saving")

    # log parameters
    parser.add_argument("--log_tensorboard", action='store_false', default=True, 
                        help='log training data in tensorboard')
    parser.add_argument("--log_communication", action='store_true', default=False)

    # eval parameters
    parser.add_argument("--do_eval", action='store_false', default=True, help="controls if we evaluate agents accross training.")
    parser.add_argument("--eval_interval", type=int, default=10000, help="number of steps between evaluations.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")
    parser.add_argument("--render_wait_input", default=False, action="store_true")
    parser.add_argument("--no_render", default=False, action="store_true")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="by default None. set the path to pretrained model.")

    # Intrinsic rewards parameters
    # parser.add_argument("--ir_algo", type=str, default="none")
    # parser.add_argument("--ir_mode", type=str, default="central")
    # parser.add_argument("--ir_coeff", type=float, default=1.0)
    # parser.add_argument("--ir_enc_dim", type=int, default=32)
    # parser.add_argument("--ir_lr", type=float, default=1e-4)
    # parser.add_argument("--ir_hidden_dim", type=int, default=64)
    # parser.add_argument("--ir_scale_fac", type=float, default=0.5)
    # parser.add_argument("--ir_ridge", type=float, default=0.1)
    # parser.add_argument("--ir_ablation", type=str, default=None)

    # LMC parameters
    parser.add_argument("--context_dim", type=int, default=16)
    parser.add_argument("--comm_head_learns_rl", default=True, 
                        action="store_false")

    # Language Learning parameters
    parser.add_argument("--lang_hidden_dim", type=int, default=32)
    parser.add_argument("--lang_lr", type=float, default=0.007)
    parser.add_argument("--lang_n_epochs", type=int, default=2)
    parser.add_argument("--lang_batch_size", type=int, default=128)

    # Communication parameters
    # parser.add_argument("--comm_policy_type", type=str, default="perfect_comm",
    #                     choices=["context_mappo", "perfect_comm", "no_comm"])
    # parser.add_argument("--comm_hidden_dim", type=int, default=64)
    # Communication ppo parameters
    # parser.add_argument("--comm_policy_algo", type=str, default='mappo', 
    #                     choices=["mappo", "rmappo", "ippo"])
    # parser.add_argument("--comm_lr", type=float, default=0.0005)
    # parser.add_argument("--comm_gamma", type=float, default=0.99)
    # parser.add_argument("--comm_ppo_epochs", type=int, default=16)
    # parser.add_argument("--comm_n_warmup_steps", type=int, default=100000)
    # parser.add_argument("--comm_ppo_clip_param", type=float, default=0.2)
    # parser.add_argument("--comm_entropy_coef", type=float, default=0.01)
    # parser.add_argument("--comm_vloss_coef", type=float, default=0.5)
    # parser.add_argument("--comm_max_grad_norm", type=float, default=10.0)
    # parser.add_argument("--comm_num_mini_batch", type=int, default=2)
    # Message generation
    parser.add_argument("--comm_max_sent_len", type=int, default=12)
    parser.add_argument("--comm_train_topk", type=int, default=1, 
                        help="k value for top-k sampling during training.")
    # Communication evaluation
    parser.add_argument("--comm_token_penalty", type=float, default=0.1)
    # parser.add_argument("--comm_klpretrain_coef", type=float, default=0.01)
    parser.add_argument("--comm_env_reward_coef", type=float, default=1.0)
    # Communication evaluation (context)
    # parser.add_argument("--comm_obs_dist_coef", type=float, default=0.1)
    # parser.add_argument("--comm_shared_mem_coef", type=float, default=1.0)
    # parser.add_argument("--comm_shared_mem_reward_type", type=str, 
    #                     choices=["direct", "shaping"], default="direct")
    parser.add_argument("--comm_noreward_empty_mess", default=False, 
                        action="store_true")

    # Shared Memory parameters
    # parser.add_argument("--use_shared_mem", action='store_true', default=False, 
    #                     help="Whether to use the shared memory, will also produce a global state of the environment at each step, in addition to the observations.")
    # parser.add_argument("--shared_mem_hidden_dim", type=int, default=64)
    # parser.add_argument("--shared_mem_n_rec_layers", type=int, default=1)
    # parser.add_argument("--shared_mem_lr", type=float, default=0.0005)
    # parser.add_argument("--shared_mem_max_buffer_size", type=int, default=1000, 
    #                     help="Max number of episodes (=*ep_length steps) stored in the buffer.")
    # parser.add_argument("--shared_mem_batch_size", type=int, default=16, 
    #                     help="Number of episodes (=*ep_length steps) sampled for a single training update.")

    # MA_GYM parameters
    parser.add_argument("--magym_n_agents", type=int, default=4)
    parser.add_argument("--magym_env_size", type=int, default=7)
    parser.add_argument("--magym_n_preys", type=int, default=2)

    # Fine-tune parameters
    parser.add_argument("--FT_pretrained_model_path", type=str, default=None)
    parser.add_argument("--FT_n_steps_fix_policy", type=int, default=10000)

    return parser
