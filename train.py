from distutils.util import strtobool
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
import gymnasium as gym
from .algos.ppo.agent import Agent
from env import make_env
import yaml
from .algos.ppo.ppo import ppo


def train_agent(envs, agent):
    # state = envs.reset(seed=0)
    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    # print(current_working_directory)
    with open('train/planner/algos/ppo/config.yaml', "r") as f:
        config_ = yaml.safe_load(f)
    # run_name = f"{config_['gym_id}__{config_['exp_name}__{config_['seed}__{int(time.time())}"
    run_name = f"{config_['gym_id']}__{config_['exp_name']}"

    # Weights and Biases
    if config_['track']:
        import wandb

        wandb.init(
            project=config_['wandb_project_name'],
            entity=config_['wandb_entity'],
            sync_tensorboard=True,
            config=config_,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Testing tensorboard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %('\n'.join([f"|{key}|{value}|" for key, value in config_.items()])),
    )

    # Seeding
    random.seed(config_['seed'])
    np.random.seed(config_['seed'])
    torch.manual_seed(config_['seed'])
    torch.backends.cudnn.deterministic = config_['torch_det']

    device = torch.device("cuda" if torch.cuda.is_available() and config_['cuda'] else "cpu")
    
    # # Env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(config_['gym_id'], config_['seed'] + i, i, config_['capture_video'], run_name) for i in range(config_['num_envs'])]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only descrete action space is supported"

    # print(envs.single_action_space.shape)
    # print(envs.observation_space)

    # agent = Agent(envs, input_size=(130, 100)).to(device)

    trained_agent, envs = ppo(envs, agent, config_, device, writer)

    envs.close()
    writer.close()

    # Save trained model
    torch.save(trained_agent.state_dict(), f"runs/{run_name}/trained_agent.pt")
    
