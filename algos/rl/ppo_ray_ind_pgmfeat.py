import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

ray.init()
env_configs = {}

num_pgms = 1
from gym_hls.envs.chstone_bm import get_chstone, get_others, get_all9
bms = get_all9()
#bms = bms[0:1]
for i, bm in enumerate(bms):
  pgm, path = bm
  env_configs['pgm'] = pgm
  env_configs['pgm_dir'] = path
  env_configs['run_dir'] = 'run_'+str(i)
  env_configs['verbose'] = True
  env_configs['log_results'] = True

  print("Tune for {}".format(pgm))
  tune.run_experiments({
      "my_experiment": {
          "run": "PPO",
          "env":HLSEnv,
          #"checkpoint_freq": 4,
          "stop": {"episodes_total": 200},
          "config": {
              "sample_batch_size": 3,
              "train_batch_size": 10,
              "sgd_minibatch_size": 3,
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
