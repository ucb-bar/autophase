#!/usr/bin/env python
#
# Optimize blocksize of apps/mmm_block.cpp
#
# This is an extremely simplified version meant only for tutorials
#
from __future__ import print_function
#import adddeps  # fix sys.path

from gym_hls.envs.hls_env import HLSEnv

import pickle

if __name__ == '__main__':

  from gym_hls.envs.random_bm import get_random
  num_pgms = 200
  bms = get_random(N=num_pgms)
  bms = bms[10:110]
  algo_pass = {}
  #algo_pass['autotuner'] = [18, 7, 37, 31, 38, 17, 2, 6, 23, 21, 24, 41, 19, 20, 33, 11, 44, 26, 4, 34, 27, 32, 8, 22, 36]
  #algo_pass['ga'] = [34, 0, 0, 0, 20, 1, 41, 0, 23, 12, 0, 0, 34, 0, 33, 0, 0, 17, 27, 15, 0, 1, 0, 0, 0, 23, 7, 0, 29, 31, 33, 0, 10, 36, 0, 0, 28, 36, 0, 5, 13, 0, 1, 0, 26] 
  #algo_pass['ga'] = [0, 1, 43, 1, 1, 0, 25, 1, 23, 16, 0, 0, 0, 33, 0, 18, 16, 0, 37, 0, 35, 0, 1, 0, 1, 14, 14, 0, 23, 1, 44, 0, 1, 33, 1, 1, 28, 0, 1, 0, 0, 0, 0, 1, 2]
  algo_pass['greedy'] = [23, 20, 34, 11, 33, 10, 32]

  for algo, passes in algo_pass.items(): 

    fout = open(algo+".csv", "w")
    for i, bm in enumerate(bms):
      pgm, files= bm
      env_configs = {}
      env_configs['pgm'] = pgm 
      env_configs['pgm_files'] = files
      env_configs['run_dir'] = 'run_'+pgm.replace(".c","")
      #env_configs['feature_type'] = 'act_hist'
      env_configs['verbose'] = True
      env_configs['log_results'] = True

      env = HLSEnv(env_configs)
      cycle = env.get_cycles(passes) 
      print("{}, {}".format(pgm.replace(".c",""), cycle))
      fout.write("{}, {}\n".format(pgm.replace(".c",""), cycle))
    fout.close()

