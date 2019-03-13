# Returns list of (pgm, pmg_files) tuple
from gym_hls.envs.utils import lsFiles
from os.path import isfile, join

def get_random(path="/scratch/qijing.huang/random_pgm/dataset", N=None):
    pgms = lsFiles(path, with_dir=False)
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
