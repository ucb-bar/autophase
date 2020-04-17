# Returns list of (pgm, pmg_files) tuple
from gym_hls.envs.utils import lsFiles
from os.path import isfile, join
import pickle

def get_random(path="/scratch/qijing.huang/random_pgm/dataset", pkl_path='/scratch/qijing.huang/AutoPhase/gym-hls/gym_hls/envs/', N=None, pgm_list=None):
  """
  Examples :
    >>> print(random_bm([“blob”]))
    (“blob.c”, [ “/scratch/qijing.huang/random_pgm/dataset/blob.c", “/scratch/qijing.huang/random_pgm/dataset/skeleton/fl.txt”] )

  Args:
    path (str, optional): path of the directory that contains the benchmarks we are interested in. Defaults to "/scratch/qijing.huang/random_pgm/dataset".
    pkl_path (str, optional): pkl_path is the path of the gym_hls/envs directory. Defaults to '/scratch/qijing.huang/AutoPhase/gym-hls/gym_hls/envs/'.
    N (int, optional): N is the number of programs from the pgm_list to chose from.
    pgm_list (list, optional): pgm_list is a list of strings that contain programs (benchmarks).
  Returns:
    Returns a list of tuples where the first element of the tuple contains the name of the program and the second element of the tuple contains the path 
    (the given parameter path + program_name) of the program and the path to the files inside the skeleton directory.
  """

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
