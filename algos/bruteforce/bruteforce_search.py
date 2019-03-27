from env import Env
import numpy as np
import getcycle
from itertools import permutations
import time
from multiprocessing.dummy import Pool as ThreadPool

def int2base(x, base, length=6):
  indices = np.zeros(length, dtype=int)
  i = 0
  while x:
    indices[i] = int(x % base)
    x = int(x / base)
    i+=1

  return indices.tolist()

def complete_walk(env, length=6):
  #env = Env(pgm)  
  ind = []
  all_passes = getcycle.qw(getcycle.opt_passes_str)
  total_passes = len(all_passes)
  
  #sequences = np.zero(total_passes)
  best_seq = []
  best_reward = -10000000

  total_num = total_passes ** length
  for i in range(total_num):
    indices = int2base(i , total_passes, length)
    _, reward = env.reset(init=indices)
    #print(i)
    if reward > best_reward:
      best_reward = reward
      best_seq = indices
      print("At cycle %d"%i)
      print("New Best Reward for %s: %d"%(env.pgm, best_reward))
      print("New Best Seq for %s: %s "%(env.pgm, best_seq))
     
  return (best_reward, best_seq)

PASS_LEN=3
def bruteforce_search(env):
  begin = time.time()
  best_reward, best_seq = complete_walk(env, PASS_LEN)
  end = time.time()
  print("Final Best Reward for %s: %d"%(env.pgm, best_reward))
  print("Final Best Seq for %s: %s "%(env.pgm, best_seq))
  print("Duration: {}".format(int(end- begin)))
  return (env.pgm, best_reward, best_seq)
  
if __name__ == "__main__":

  from chstone_bm import get_chstone, get_others
  bm = get_chstone()
  bm.extend(get_others())
  envs = []
  for pgm, path in bm:
    env=Env(pgm, path, "run_"+pgm)  
    envs.append(env)

  num_threads = len(envs)
  pool = ThreadPool(num_threads)
  results = pool.map(bruteforce_search, envs)
  print(results)

