from __future__ import print_function
#import adddeps  # fix sys.path

from gym_hls.envs.hls_env import HLSEnv

import pickle

if __name__ == '__main__':
  from gym_hls.envs.chstone_bm import get_chstone, get_others, get_all9
  bms = get_all9()

  algo_pass = {}
  #algo_pass['autotuner'] = [18, 7, 37, 31, 38, 17, 2, 6, 23, 21, 24, 41, 19, 20, 33, 11, 44, 26, 4, 34, 27, 32, 8, 22, 36]
  #algo_pass['ga'] = [0, 1, 43, 1, 1, 0, 25, 1, 23, 16, 0, 0, 0, 33, 0, 18, 16, 0, 37, 0, 35, 0, 1, 0, 1, 14, 14, 0, 23, 1, 44, 0, 1, 33, 1, 1, 28, 0, 1, 0, 0, 0, 0, 1, 2]
  #algo_pass['greedy'] = [23, 20, 34, 11, 33, 10, 32]
  #algo_pass['greedy'] = [23, 20, 34, 11, 33, 10, 32,  9, 31,  8, 30,  7, 29,  6, 28,  5, 27]
    
  # Train on random 
  #algo_pass['greedy'] =  [10,  8, 15, 11,  6, 13, 10,  7, 11,  3, 13,  7,  7,  7,  7,  7]
  #algo_pass['opentuner'] = [15, 14, 10, 8, 6, 5, 13, 7, 0] 
  #algo_pass['ga'] = [4, 1, 0, 8, 11, 1, 2, 15, 13, 11, 11, 2, 1, 4, 13, 1]
  algo_pass['ga'] = [13, 9, 15, 10, 14, 7, 5, 14, 15, 6, 0, 11, 13, 4, 15, 4]

  for algo, passes in algo_pass.items(): 

    fout = open(algo+".csv", "w")
    for i, bm in enumerate(bms):
      pgm, files= bm
      env_configs = {}
      env_configs['pgm'] = pgm 
      env_configs['pgm_dir'] = files
      env_configs['run_dir'] = 'run_'+pgm.replace(".c","")
      #env_configs['feature_type'] = 'act_hist'
      env_configs['verbose'] = True
      env_configs['log_results'] = True
      env_configs['shrink'] = True

      env = HLSEnv(env_configs)
      cycle = env.get_cycles(passes) 
      print("{}, {}".format(pgm.replace(".c",""), cycle))
      fout.write("{}, {}\n".format(pgm.replace(".c",""), cycle))
    fout.close()

