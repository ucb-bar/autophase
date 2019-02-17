import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

ray.init()
env_configs = {}
 
tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        "env":HLSMultiEnv,
        "stop": {"episode_reward_mean": 0},
        "config": {
            "sample_batch_size": 100,
            "train_batch_size": 700,
            "sgd_minibatch_size": 70,
            "horizon": 10,
            "num_gpus": 2,
            "num_workers": 5,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "env_config": env_configs, 
        },
    },
})
