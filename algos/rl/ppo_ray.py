import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

ray.init()
env_configs = {}

num_pgms = 1
from gym_hls.envs.chstone_bm import get_chstone, get_others
bms = get_chstone(N=num_pgms)
for i, bm in enumerate(bms):
  pgm, path = bm
  env_configs['pgm'] = pgm
  env_configs['pgm_dir'] = path
  env_configs['run_dir'] = 'run_'+str(i)

tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        "env":HLSEnv,
        "checkpoint_freq": 4,
        "stop": {"episode_reward_mean": 0},
        "config": {
            "sample_batch_size": 100,
            "train_batch_size": 700,
            "sgd_minibatch_size": 70,
            "horizon": 12,
            "num_gpus": 2,
            "num_workers": 7,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "env_config": env_configs,
        },
    },
})
