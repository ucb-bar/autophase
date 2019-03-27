import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

ray.init()
env_configs = {}

from gym_hls.envs.random_bm import get_random
num_pgms = 1000
bms = get_random(N=num_pgms)
for i, bm in enumerate(bms):
  pgm, files= bm
  env_configs['pgm'] = pgm
  env_configs['pgm_files'] = files
  env_configs['run_dir'] = 'run_'+pgm.replace(".c","")
  #env_configs['feature_type'] = 'act_hist'
  env_configs['verbose'] = True
  env_configs['log_results'] = True

  print("Tune for {}".format(pgm))
  tune.run_experiments({
      "my_experiment": {
          "run": "PPO",
          "env":HLSEnv,
          "checkpoint_freq": 50,
          "stop": {"episodes_total": 500},
          "config": {
              "sample_batch_size": 10,
              "train_batch_size": 100,
              "sgd_minibatch_size": 8,
              "num_sgd_iter": 10,
              "horizon": 12,
              "num_gpus": 1,
              "num_workers": 5,
              "lr": 1e-3,
              #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
              "vf_clip_param": 1e5,
              "env_config": env_configs,
          },
      },
  })
