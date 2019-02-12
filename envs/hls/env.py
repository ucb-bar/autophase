import getcycle
import os
import getfeatures
import datetime
import glob
import shutil


class Env(object):
  def __init__(self, pgm, pgm_dir="./examples", run_dir=None, delete_run_dir=True, init_with_passes=True):

    self.delete_run_dir = delete_run_dir
    if run_dir:
      self.run_dir = run_dir
    else:
      currentDT = datetime.datetime.now()
      self.run_dir ="run-"+currentDT.strftime("%Y-%m-%d-%H-%M-%S-%f")

    cwd = os.getcwd()
    self.run_dir = os.path.join(cwd, self.run_dir)
    #os.mkdir(self.run_dir)

    if os.path.isdir(self.run_dir):
      shutil.rmtree(self.run_dir)

    shutil.copytree(pgm_dir, self.run_dir)
   # files = []
   # files.extend(list(glob.iglob(os.path.join(pgm_dir, "*.c"))))
   # files.extend(list(glob.iglob(os.path.join(pgm_dir, "*.h"))))
   # for f in files:
   #   if os.path.isfile(f):
   #     #print(f)
   #     shutil.copy2(f, self.run_dir) 

    self.pre_passes_str= "-prune-eh -functionattrs -ipsccp -globalopt -mem2reg -deadargelim -sroa -early-cse -loweratomic -instcombine -loop-simplify"
    self.pre_passes = getcycle.passes2indice(self.pre_passes_str)
    self.passes = []
    if init_with_passes:
      self.passes.extend(self.pre_passes)
    self.pgm = pgm
    self.pgm_name = pgm.replace('.c','') 
    self.bc = self.pgm_name + '.prelto.2.bc'
    self.delete_run_dir = delete_run_dir

  def __del__(self):
    if self.delete_run_dir:
      if os.path.isdir(self.run_dir):
        shutil.rmtree(self.run_dir)

  def get_O3_rewards(self, sim=False):
    #os.chdir(self.run_dir)
    from geto3 import getO3Cycles
    cycle = getO3Cycles(self.pgm_name, self.run_dir, sim=sim)
    return -cycle
    #os.chdir("..")

  def get_rewards(self, sim=False):
    cycle = getcycle.getHWCycles(self.pgm_name, self.passes, self.run_dir, sim=sim)
    #print("passes: {}, rw: {}".format(self.passes, -cycle))
    return -cycle

  def get_obs(self):
    #os.chdir(self.run_dir)
    feat = getfeatures.run_stats(self.bc, self.run_dir)
    #os.chdir("..")
    return feat

  # reset() resets passes to []
  # reset(init=[1,2,3]) resets passes to [1,2,3]
  def reset(self, init=None,init_with_passes=True, get_obs=True, ret=True, sim=False):

    self.passes = []
    if init_with_passes:
      self.passes.extend(self.pre_passes)

    if init:
      self.passes.extend(init)

    if ret:
      reward = self.get_rewards(sim=sim)
      obs = []
      if get_obs:
        obs = self.get_obs()

      return (obs, reward)
    else:
      return 0

  def step(self, action, get_obs=True):
    self.passes.append(action)
    reward = self.get_rewards()
    obs = []
    if get_obs:
      obs = self.get_obs()
    return (obs, reward)
    
  def multi_steps(self, actions):
    self.passes.extend(actions)
    return (self.get_obs(), self.get_rewards())


def getO3():
  import time
  from chstone_bm import get_chstone, get_others
  bm = get_chstone()
  bm.extend(get_others())
  fout = open("report_O3"+".txt", "w")
  fout.write("Benchmark |Cycle Counts | Algorithm Runtime (s)|Passes \n")


  for pgm, path in bm:
    print(pgm)
    begin = time.time()
    
    env=Env(pgm, path, delete_run_dir=True, init_with_passes=True)  
    cycle = - env.get_O3_rewards()
    end = time.time()
    compile_time = end - begin
    fout.write("{}|{}|{}|{}\n".format(pgm, cycle, compile_time, "-O3"))

def test():
  from chstone_bm import get_chstone, get_others
  import numpy as np
  bm = get_chstone(N=4)

  envs = []
  i = 0
  for pg, path in bm:
      envs.append(Env(pg, path, "run_env_"+str(i)))
      i = i+1

  #test_passes = [23,31,33] 
  test_passes = [0, 12, 23]
  from multiprocessing.pool import ThreadPool
  pool = ThreadPool(len(envs))
  rews = pool.map(lambda env: env.reset(init=test_passes, get_obs=False)[1], envs)
  print(rews)
  def geo_mean(iterable):
    a = np.array(iterable).astype(float)
    prod = a.prod()
    prod = -prod if prod < 0 else prod
    return prod**(1.0/len(a))

  print(geo_mean(rews)) 

  
def main():
  #env=Env("adpcm.c", "/scratch/qijing.huang/LegUp/legup-4.0/examples/chstone/adpcm", delete_run_dir=False)  
  env=Env("gsm.c", "./examples", delete_run_dir=True)  
  env.reset(init_with_passes=True)
  #obs, rwd =env.reset(init_with_passes=False)
  #print(obs)
  #obs, rwd = env.reset(init=[7, 16, 25])
  #passes=[0, 12, 23]
  #obs, rwd = env.reset(init=passes)
  #print(rwd)
  #return

  from chstone_bm import get_bms
  bm = get_bms("all12")
  #bm = get_bms("gsm")
  
  rwds = []
  tup = []
  for pgm, path in bm:
    print(pgm)
    #env=Env(pgm, path, run_dir="./test_sim", delete_run_dir=True, init_with_passes=True)  
    env=Env(pgm, path, run_dir="run_sim", delete_run_dir=False, init_with_passes=True)  
    #obs, rwd =env.reset(init_with_passes=True)
    pg = [30, 31, 41, 23, 4, 8, 27, 0, 33, 13, 38, 1]
    pg_rtg = [30, 31, 41, 23, 4, 8, 27, 0, 33, 13, 38, 1]
    ga = [0, 0, 0, 0, 0, 23, 27, 27, 33, 31, 24, 10]
    pg = [30, 31, 41, 23, 15, 8, 2, 0, 33, 13, 39, 2]
    dqn_12 = [16, 25, 6, 10, 23, 22, 18, 11, 31, 34, 27, 33]
    pg =[25, 33, 23, 33, 31, 23, 33, 23, 33, 23, 33, 33]
    pg_24=[8, 31, 5, 23, 37, 37, 23, 37, 21, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 23, 37]
    pg_24_b100=[23, 8, 32, 24, 8, 20, 9, 33, 30, 37, 30, 25, 30, 13, 31, 31, 3, 21, 31, 18, 15, 25, 15, 30]
    #pg_24_b200=[31, 18, 9, 14, 11, 16, 6, 34, 26, 34, 4, 15, 0, 30, 28, 1, 23, 44, 5, 5, 36, 19, 31, 13]
    pg_24_b200=[25, 31, 24, 31, 31, 33, 23, 31, 33, 31, 38, 31, 33, 31, 31, 23, 33, 33, 31, 33, 33, 31, 33, 33]
    in_24=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 27, 27, 33, 31, 24, 10]
    #pg_24_b500=[3, 44, 20, 30, 31, 28, 12, 31, 31, 31, 15, 27, 25, 38, 23, 21, 38, 9, 21, 33, 23, 38, 31, 33]
    pg_24_b500=[30, 37, 16, 24, 4, 43, 25, 23, 18, 31, 16, 5, 42, 43, 23, 18, 23, 43, 27, 10, 15, 33, 30, 6]
    ga_24 = [43, 20, 25, 30, 2, 23, 0, 44, 2, 0, 0, 25, 9, 10, 0, 1, 13, 23, 31, 0, 11, 0, 33, 24]
    pg_ot_b100=[28, 12, 6, 2, 1, 23, 33, 21, 23, 16, 6, 27]
    pg_ot_b200=[28, 12, 7, 2, 7, 23, 33, 21, 2, 2, 30, 27]
    pg_ot_b500=[10, 11, 23, 12, 8, 31, 26, 23, 41, 25, 33, 5]
    pg_b100_7=[43, 31, 23, 23, 38, 38, 23, 44, 19, 23, 33, 33]
    pg_b100_8=[37, 24, 25, 0, 32, 26, 29, 31, 23, 31, 8, 37]
    pg_b200_4=[0, 41, 23, 41, 33, 14, 35,     0, 1, 14, 21, 27]
   
    fir=[14, 22, 23]
    fir=[23, 9, 31, 0, 25, 30]
    dhry=[5, 20, 43, 18, 16, 36, 33, 44, 41, 23]

    pg_b100_5=[25, 23, 8, 23, 22, 42, 38, 27, 36, 25, 23, 33]
    #pg_b100_5_200MHz=[30, 25, 32, 9, 0, 8, 9, 9, 23, 25, 23, 33]
    #obs, rwd =env.reset(init=pg_b100_5_200MHz,init_with_passes=True, sim=True)
    obs, rwd =env.reset(init=pg_b100_5,init_with_passes=True)
    #rwd =env.get_O3_rewards(sim=True)
    #rwd =env.get_O3_rewards()
    #obs, rwd =env.reset(init=pg_b100_5,init_with_passes=True)
    #obs, rwd =env.reset(init=dqn_12,init_with_passes=True)
    #print(obs)
    rwds.append(rwd)
    print(rwd)
    tup.append((pgm.replace(".c",""), -rwd))
    #env.step(4)
  from geomean import geomean
  print(geomean(rwds))

  for pgm, rwd in sorted(tup):
    print("{}, {}".format(pgm, rwd))
  

if __name__ == "__main__":
  main()
  #getO3()
  #test()

