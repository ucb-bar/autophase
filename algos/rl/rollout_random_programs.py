from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv
import os
import pickle
env_configs = {}
import argparse
NumSteps = 12
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str, required=True)
parser.add_argument('--steps', '-s', type=int, default=NumSteps)
args = parser.parse_args()

from gym_hls.envs.random_bm import get_random
bms = get_random(N=1000)
bms = bms[-10:]

needed_config_for_rollout = {}
for i, bm in enumerate(bms):
   pgm, files = bm
   env_configs["pgm"] = pgm
   env_configs["verbose"] = 'True'
   env_configs["pgm_files"] = files
   env_configs["run_dir"] = 'run_'+pgm.replace(".c","")
   env_configs["orig_and_normalize"] = 'True'
   needed_config_for_rollout["env_config"] = env_configs
   #needed_config_for_rollout["model"] = {"use_lstm": True, "max_seq_len":5, "lstm_use_prev_action_reward":True}
   #needed_config_for_rollout["model"]= {"fcnet_hiddens":[256,256,256,256]}
   str_env_config = str(needed_config_for_rollout).replace("\'","\"")
   #print(str_env_config)
   print('-'*30)
   command = "rllib rollout {0} --run PPO --env HLS-v0 --steps {1} --no-render --config '{2}'".format(args.           checkpoint_dir,args.steps,str_env_config)
   print("running ", command)
   print('-'*30)
   os.system(command)

   results = pickle.load(open('cycles.pkl','rb'))
   with open('results.csv','w') as f:
      for key in results.keys():
          f.write("%s,%s\n"%(key,results[key]))


