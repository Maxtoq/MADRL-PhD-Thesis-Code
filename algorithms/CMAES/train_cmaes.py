import argparse
import torch
import cma
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import get_paths, load_scenario_config, make_env


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=32, nb_hidden_layers=0, 
                 linear=False, nonlin=F.relu, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nb_hidden_layers (int): Number of hidden layers
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(PolicyNetwork, self).__init__()

        if linear:
            self.fc_in = nn.Linear(input_dim, out_dim)
        else:
            self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = []
        for i in range(nb_hidden_layers):
            self.fc_hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.linear = linear
        if not discrete_action:
            # Constrain between 0 and 1
            # initialize small to prevent saturation
            self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # one hot argmax
            self.out_fn = lambda x: (x == x.max(1, keepdim=True)[0]).float()

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions)
        """
        x = self.nonlin(self.fc_in(X))
        if not self.linear:
            for fc in self.fc_hidden:
                x = self.nonlin(fc(x))
            x = self.fc_out(x)
        out = self.out_fn(x)
        return out


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def load_array_in_model(param_array, model):
    new_state_dict = model.state_dict()
    for key, value in new_state_dict.items():
        size = np.prod(value.shape)
        layer_params = param_array[:size]
        param_array = param_array[size:]
        param_tensor = torch.from_numpy(layer_params.reshape(value.shape))
        new_state_dict[key] = param_tensor
    model.load_state_dict(new_state_dict, strict=True)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def run(config):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(config)
    print("Saving model in dir", run_dir)

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(config, run_dir)
    nb_agents = sce_conf['nb_agents']
    
    # Initiate env
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    env = make_env(config.env_path, sce_conf, 
                   discrete_action=config.discrete_action)

    # Create model
    num_in_pol = env.observation_space[0].shape[0]
    if config.discrete_action:
        num_out_pol = env.action_space[0].n
    else:
        num_out_pol = env.action_space[0].shape[0]
    policy = PolicyNetwork(num_in_pol, num_out_pol, config.hidden_dim,
                           linear=config.linear, 
                           discrete_action=config.discrete_action)
    policy.eval()

    # Create the CMA-ES trainer
    es = cma.CMAEvolutionStrategy(np.zeros(get_num_params(policy)), 1, 
                                            {'seed': config.seed})
    print('Pop_size =', es.popsize)
    
    # for ep_i in tqdm(range(0, 
    #                        config.n_episodes, 
    #                        es.popsize * config.n_eps_per_eval)):
    for ev_i in tqdm(range(0, config.n_evals)):
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor

        # Ask for candidate solutions
        solutions = es.ask()

        # Initialize seed for this round of evaluations
        seed = np.random.randint(1e9)

        # Perform one episode for each solution
        tell_rewards = []
        for sol_i in range(len(solutions)):
            # Load solution in model
            load_array_in_model(solutions[sol_i], policy)
            
            # Seed eval
            np.random.seed(seed)

            # Perform n_eps_per_eval episodes and average the rewards
            sol_rewards = []
            for eval_i in range(config.n_eps_per_eval):
                # Reset env
                obs = env.reset()
                episode_reward = 0.0
                for et_i in range(config.episode_length):
                    # Rearrange observations to fit in the model
                    torch_obs = Variable(torch.Tensor(np.vstack(obs)),
                                        requires_grad=False)
                
                    actions = policy(torch_obs)

                    # Convert actions to numpy arrays
                    agent_actions = [ac.data.numpy() for ac in actions]

                    next_obs, rewards, dones, infos = env.step(agent_actions)

                    episode_reward += sum(rewards) / nb_agents

                    if dones[0]:
                        break

                    obs = next_obs
                sol_rewards.append(-episode_reward)
            # Store average rewards of current solution
            tell_rewards.append(sum(sol_rewards) / config.n_eps_per_eval)

        # Update CMA-ES model
        es.tell(solutions, tell_rewards)

        # Get id of best element in solutions
        best_sol_i = np.argmin(tell_rewards)

        # Log rewards
        logger.add_scalar('cmaes/mean_episode_rewards', 
                          -sum(tell_rewards) / es.popsize, ev_i)

        # Save model
        if ev_i % config.save_interval < es.popsize:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            # Load best solution in model
            load_array_in_model(solutions[best_sol_i], policy)
            save_model(policy, run_dir / 'incremental' / 
                                ('model_ep%i.pt' % (ev_i + 1)))
            save_model(policy, model_cp_path)

    # Load best solution in model
    load_array_in_model(solutions[best_sol_i], policy)
    save_model(policy, model_cp_path)   
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    print("Model saved in dir", run_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_evals", default=3500, type=int)
    #parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--save_interval", default=50, type=int)
    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--n_eps_per_eval", default=1, type=int)

    config = parser.parse_args()

    run(config)
