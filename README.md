## Push Particle Environment
Code for a 2D environment where agents need to push objects to put them on landmarks. 

## Multiagent Particle Environment
The code uses the [Multiagent Particle Environment](https://github.com/openai/multiagent-particle-envs). To use it, follow the installation instructions.

## Run training
To train a model with the `train.py` script, run the following command:
> python train.py <scenario_script_path> <model_name>

This command will run the training script for the MADDPG model. The first argument is the python script that contains the scenario to train on. The second is the name of the model for saving the logs and checkpoint.

The default number of episodes is 25000, which is only good for checking that the script works. For a proper training process, you have to specify the number of episodes with the argument `--n_episodes`. You'll also need to set the number of exploration episodes with the argument `--n_exploration_eps`, otherwise it will be set to the default 25000.

A classic command for running the training script on the pushing scenario:
> python train.py coop_push_scenario.py maddpg --n_episodes=1000000 --n_exploration_eps=1000000

Command for running the training script on the coop push scenario with 30 parallel environments, with discrete actions and a special scenario config file:
> python train.py coop_push_scenario.py 2addpg_BIG_fo_disc_abs_distrew_100 --n_episodes=1000000 --n_exploration_eps=1000000 --n_rollout_threads=30 --steps_per_update=30000 --batch_size=4096 --discrete_action --sce_conf_path configs/2a_1o_fullobs_absolute_distrew.json

## Continue an existing run
You can continue an existing run, loading a checkpoint for the model. To do so, use the argument `--run_name`. The run name must be of the form "run<number>" and correspond to an existing directory where a run has been saved. This directory should therefore contain a "model.pt" file. 