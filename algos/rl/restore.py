import argparse
import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()
ray.init()

env_config = {
        'verbose': True,
        'feature_type':'act_pgm'
        }
config_restore = {
            "sample_batch_size": 50,
            "train_batch_size": 200,
            "sgd_minibatch_size": 40,
            #"model": {"use_lstm": True},
            "horizon": 45,
            "num_gpus": 2,
            "num_workers": 7,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "env_config": env_config,
        }
tune.run_experiments({
         "restore_ppo": {
         "run": "PPO",
         "env": HLSMultiEnv,
         "restore": args.checkpoint_dir,
         "checkpoint_freq":10,
         "config": config_restore
         },
})
