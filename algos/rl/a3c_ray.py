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
  pgm, path = bm
  env_configs['pgm'] = pgm
  env_configs['pgm_dir'] = path
  env_configs['run_dir'] = 'run_a3c_'+pgm.replace(".c","")
  #env_configs['feature_type'] = 'act_hist'
  env_configs['verbose'] = True
  env_configs['log_results'] = True

  tune.run_experiments({
      "a3c_jenny": {
          "run": "A3C",
          "env":HLSEnv,
          "checkpoint_freq": 10,
          "stop": {"episodes_total": 50},
          "config": {

      # === Model ===
    "num_workers": 2,
    "horizon" : 45,
    # Size of rollout batch
    "sample_batch_size": 10,
    # Use PyTorch as backend - no LSTM support
    "use_pytorch": False,
    # GAE(gamma) parameter
    "lambda": 1.0,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rate
    "lr": 0.0001,
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5,
    # Entropy coefficient
    "entropy_coeff": 0.01,
    # Min time per iteration
    "min_iter_time_s": 5,
    # Workers sample async. Note that this increases the effective
    # sample_batch_size by up to 5x due to async buffering of batches.
    "sample_async": True,
    "env_config": env_configs,
          },
      },
  })
