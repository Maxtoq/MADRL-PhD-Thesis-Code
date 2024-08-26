import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='lmc', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--experiment_name", type=str, default="TEST", 
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda_device", default=None, type=str)
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, 
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_parallel_envs", type=int, default=1,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_steps", type=int, default=int(10e6),
                        help='Number of environment steps to train (default: 10e6)')

    # env parameters
    parser.add_argument("--env_name", type=str, default='magym_PredPrey', 
                        help="specify the names of environment and the task")
    parser.add_argument("--episode_length", type=int,
                        default=100, help="Max length for any episode")

    # replay buffer parameters
    parser.add_argument("--rollout_length", type=int,
                        default=100, help="Number of steps done during each rollout phase.")
    parser.add_argument("--n_mini_batch", type=int, default=2,
                        help='number of batches for ppo (default: 1)')

    # network parameters
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--policy_layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")

    # recurrent parameters
    parser.add_argument("--policy_recurrent_N", type=int, default=1, 
                        help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--n_warmup_steps", type=int, default=50000)
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    parser.add_argument("--share_params", default=False, action='store_true')

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1000000, help="number of steps between models saving")
    parser.add_argument("--save_increments", action='store_true', default=False, 
                        help='Save incremental model checkpoints throughout training.')

    # log parameters
    parser.add_argument("--log_tensorboard", action='store_false', default=True, 
                        help='log training data in tensorboard')
    parser.add_argument("--log_comm", action='store_true', default=False)

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="by default None. set the path to pretrained model.")

    # LMC parameters
    parser.add_argument("--context_dim", type=int, default=16)
    # parser.add_argument("--no_train_lang", default=False, 
    #                     action="store_true")

    # Language Learning parameters
    parser.add_argument("--lang_embed_dim", type=int, default=4)
    parser.add_argument("--lang_hidden_dim", type=int, default=32)
    parser.add_argument("--lang_lr", type=float, default=0.007)
    parser.add_argument("--lang_batch_size", type=int, default=256,
                        help="Number of steps sampled in batch for CLIP training.")
    parser.add_argument("--lang_temp", type=float, default=1.0)
    parser.add_argument("--lang_imp_sample", default=False, action="store_true")

    # Loss weights
    parser.add_argument("--dyna_weight_loss", default=False, action="store_true")
    parser.add_argument("--actor_loss_weight", type=float, default=1.0)
    parser.add_argument("--lang_capt_loss_weight", type=float, default=1.0)
    parser.add_argument("--lang_capt_loss_weight_anneal", type=float, default=1.0)

    # Communication parameters
    parser.add_argument("--comm_type", default="no_comm", 
                        choices=["language", "emergent_continuous", "no_comm",
                            "emergent_discrete", "perfect"])
    parser.add_argument("--comm_ec_strategy", default="cat", 
                        choices=["cat", "sum", "mean", "random", "nn"],
                        help="When doing emergent continuous communication, strategy for transforming incoming messages into the social context.")
    parser.add_argument("--comm_eps_smooth", type=float, default=1.0)
    # Message generation
    parser.add_argument("--comm_max_sent_len", type=int, default=12)
    parser.add_argument("--comm_train_topk", type=int, default=1, 
                        help="k value for top-k sampling during training.")
    # Communication evaluation
    parser.add_argument("--comm_token_penalty", type=float, default=0.05)
    parser.add_argument("--comm_env_reward_coef", type=float, default=1.0)
    parser.add_argument("--comm_noreward_empty_mess", default=False, 
                        action="store_true")

    # MA_GYM parameters
    parser.add_argument("--magym_n_agents", type=int, default=4)
    parser.add_argument("--magym_env_size", type=int, default=7)
    parser.add_argument("--magym_n_preys", type=int, default=2)
    parser.add_argument("--magym_obs_range", type=int, default=5)
    parser.add_argument("--magym_no_purple", default=False, action="store_true")
    parser.add_argument("--magym_see_agents", default=False, action="store_true")


    # Fine-tuning parameters
    parser.add_argument("--FT_env_name", type=str, default=None)
    parser.add_argument("--FT_magym_env_size", type=int, default=None)
    parser.add_argument("--FT_magym_actual_obsrange", type=int, default=None)
    parser.add_argument("--FT_freeze_lang", default=False, action="store_true")

    # eval parameters
    parser.add_argument("--do_eval", action='store_false', default=True, 
                        help="controls if we evaluate agents accross training.")
    parser.add_argument("--eval_interval", type=int, default=10000, 
                        help="number of steps between evaluations.")
    parser.add_argument("--interact", default=False, action='store_true')

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False)
    parser.add_argument("--render_episodes", type=int, default=5, 
                        help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, 
                        help="the play interval of each rendered image in saved video.")
    parser.add_argument("--render_wait_input", default=False, action="store_true")


    parser.add_argument("--continue_run", action='store_true', default=False, 
                        help="To continue a previously trained run. The argument --model_dir must also be provided.")
    parser.add_argument("--adapt_run", action='store_true', default=False, 
                        help="To adapt a previously trained run on different task. The argument --model_dir must also be provided.")

    return parser
