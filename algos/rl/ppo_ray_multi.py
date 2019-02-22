import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

ray.init()
env_configs = {'normalize':True}

tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        "env":HLSMultiEnv,
#        "checkpoint_freq": 4,
        "stop": {"episode_reward_mean": 100},
        "config": {
            "sample_batch_size": 100,
            "train_batch_size": 700,
            "sgd_minibatch_size": 70,
            "model": {"use_lstm": True, "max_seq_len":5, "lstm_use_prev_action_reward":True},
            "horizon": 24,
            "num_gpus": 0,
            "num_workers": 0,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "env_config": env_configs,
        },
    },
})
