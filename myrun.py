from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.registry import register_env
from netenv import NetworkEnv
import numpy as np
import ray

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ray.init()

gml_path = '/Users/spartan/PycharmProjects/Reinforcement/nsfnet.gml'
#registering my custom env with name "netenv-v0"

# env_instance = NetworkEnv(gml_path)

def env_creator(env_config):
    return NetworkEnv(gml_path)


# env_instance = env_creator({})
register_env("netenv-v0", env_creator)

# Set up RL

# Algo 1
config = (PPOConfig()
          .training(gamma=0.999, lr=0.01)
          .environment(env='netenv-v0')
          .resources(num_gpus=0)
          .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        )

# Algo 2
config = (DQNConfig()
          .training(gamma=0.999, lr=0.1,train_batch_size=32, model={"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"},)
          .environment(env='netenv-v0')
          .exploration(exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.1, "epsilon_timesteps": 10000})
          )


algo = config.build()


for _ in range(10):
    algo.train()
