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
for i, bm in enumerate(bms):
  try:
    pgm, path = bm
    env_configs['pgm'] = pgm
    env_configs['pgm_dir'] = path
    env_configs['run_dir'] = 'run_es_'+ pgm.replace(".c","")
    #env_configs['feature_type'] = 'act_hist'
    env_configs['verbose'] = True
    env_configs['log_results'] = True

    tune.run_experiments({
        "es_jenny": {
            "run": "ES",
            "env":HLSEnv,
            "checkpoint_freq": 10,
            #"stop": {"episodes_total": 200},
            "stop": {"episodes_total": 100},
            "config": {
              #"num_gpus": 2,
              "episodes_per_batch":45, 
              "train_batch_size": 50,
      "l2_coeff": 0.005,
      "noise_stdev": 0.02,
      "eval_prob": 0.06,
      "return_proc_mode": "centered_rank",
      "num_workers": 2,
      "stepsize": 0.01,
      "observation_filter": "MeanStdFilter",
      "noise_size": 250000000,
      "report_length": 10,
      "env_config": env_configs,
            },
        },
    })
  except:
    continue
