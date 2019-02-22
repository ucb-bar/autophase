from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv
import os
env_configs = {}
import argparse
NumSteps = 12
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str, required=True)
parser.add_argument('--steps', '-s', type=int, default=NumSteps)
args = parser.parse_args()

from gym_hls.envs.chstone_bm import get_chstone, get_others
bms = get_chstone()
bms.extend(get_others())
needed_config_for_rollout = {}
for i, bm in enumerate(bms):
  pgm, path = bm
  env_configs["pgm"] = pgm
  env_configs["verbose"] = "True"
  env_configs["pgm_dir"] = path
  env_configs["run_dir"] = 'run_'+str(i)
  needed_config_for_rollout["env_config"] = env_configs
  needed_config_for_rollout["model"] = {"use_lstm":"True"}
  str_env_config = str(needed_config_for_rollout).replace("\'","\"")
  #print(str_env_config)
  print('-'*30)
  command = "rllib rollout {0} --run PPO --env HLS-v0 --steps {1} --no-render --config '{2}'".format(args.checkpoint_dir,args.steps,str_env_config)
  print("running ", command)
  print('-'*30)
  os.system(command)



