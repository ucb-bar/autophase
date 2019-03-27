from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv
import os
import pickle
import csv
import argparse
NumSteps = 12

env_configs = {}
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str, required=True)
parser.add_argument('--steps', '-s', type=int, default=NumSteps)
args = parser.parse_args()

from gym_hls.envs.chstone_bm import get_chstone, get_others
bms = get_chstone()
bms.extend(get_others())
#bms = [bms[0]]
needed_config_for_rollout = {}
for i, bm in enumerate(bms):
  pgm, path = bm
  env_configs["pgm"] = pgm
  env_configs["verbose"] = "True"
  env_configs["pgm_dir"] = path
  env_configs["run_dir"] = 'run_'+str(i)
  env_configs["orig_and_normalize"] = "True"
  env_configs['log_obs_reward'] = "True"
  needed_config_for_rollout["env_config"] = env_configs
  #needed_config_for_rollout["model"]= {"fcnet_hiddens":[256,256,256,256]}
  str_env_config = str(needed_config_for_rollout).replace("\'","\"")
  #print(str_env_config)
  print('-'*30)
  command = "rllib rollout {0} --run PPO --env HLS-v0 --steps {1} --no-render --config '{2}'".format(args.checkpoint_dir,args.steps,str_env_config)
  print("running ", command)
  print('-'*30)
  os.system(command)

results = pickle.load(open('cycles.pkl','rb'))
with open('results.csv','w') as f:
    for key in results.keys():
        f.write("%s,%s\n"%(key,results[key]))



