from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

env_configs = {}

num_pgms = 1
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
  print(str(needed_config_for_rollout).replace("\'","\""))


