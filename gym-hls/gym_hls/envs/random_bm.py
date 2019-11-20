# Returns list of (pgm, pmg_files) tuple
from gym_hls.envs.utils import lsFiles
from os.path import isfile, join
import pickle

def get_random(path="/scratch/qijing.huang/random_pgm/dataset", pkl_path='/scratch/qijing.huang/AutoPhase/gym-hls/gym_hls/envs/', N=None, pgm_list=None):
  import os
  cwd = os.getcwd()
  pkl_file = open(join(pkl_path, 'random_pgms.pkl'), 'rb')
  buckets = pickle.load(pkl_file)
  pkl_file.close()

  interval = [0, 1000, 5000, 10000, 50000, 100000]
  if pgm_list is None:
    pgm_list = buckets[2] # 5000~10000
    pgms = list(map(lambda x: x[0], pgm_list))
  else: 
    pgms = list(map(lambda x: x+'.c', pgm_list))

  aux_files = lsFiles(join(path, 'skeleton'))
  random_list = []
  if N is None:
    N = len(pgms)
  for i in range(N):
    files = []
    files.append(join(path, pgms[i]))
    files.extend(aux_files)
    random_list.append((pgms[i], files))
  return random_list

print(len(get_random()))
