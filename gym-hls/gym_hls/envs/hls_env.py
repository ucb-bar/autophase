from gym_hls.envs import getcycle 
from gym_hls.envs import getfeatures
import os
import datetime
import glob
import shutil
import numpy as np
import gym
from gym.spaces import Discrete, Box

class HLSEnv(gym.Env):
  def __init__(self, env_config):
    self.action_space = Discrete(45)
    self.observation_space = Box(0.0,1.0,shape=(56,),dtype = np.float32)
    self.prev_cycles = 10000000
    
    pgm = env_config['pgm']
    pgm_dir = env_config['pgm_dir']
    run_dir = env_config.get('run_dir', None)
    delete_run_dir = env_config.get('delete_run_dir', True)
    init_with_passes = env_config.get('init_with_passes', True)
    self.delete_run_dir = delete_run_dir
    if run_dir:
      self.run_dir = run_dir+'_p'+str(os.getpid())
    else:
      currentDT = datetime.datetime.now()
      self.run_dir ="run-"+currentDT.strftime("%Y-%m-%d-%H-%M-%S-%f")+'_p'+str(os.getpid())

    cwd = os.getcwd()
    self.run_dir = os.path.join(cwd, self.run_dir)

    if os.path.isdir(self.run_dir):
      shutil.rmtree(self.run_dir, ignore_errors=True)

    shutil.copytree(pgm_dir, self.run_dir)

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
    from gym_hls.envs.geto3 import getO3Cycles
    cycle = getO3Cycles(self.pgm_name, self.run_dir, sim=sim)
    return -cycle

  def get_rewards(self, diff=True, sim=False):
    cycle, done = getcycle.getHWCycles(self.pgm_name, self.passes, self.run_dir, sim=sim)
    if (diff): 
      rew = self.prev_cycles - cycle
      self.prev_cycles = cycle
    else:
      rew = -cycle   
    print(rew)
    return rew, done

  def get_obs(self):
    feat = getfeatures.run_stats(self.bc, self.run_dir)
    return feat

  # reset() resets passes to []
  # reset(init=[1,2,3]) resets passes to [1,2,3]
  def reset(self, init=None,init_with_passes=True, get_obs=True, ret=True, sim=False):
    self.prev_cycles, _ = getcycle.getHWCycles(self.pgm_name, self.passes, self.run_dir, sim=sim)
    self.passes = []
    if init_with_passes:
      self.passes.extend(self.pre_passes)

    if init:
      self.passes.extend(init)

    if ret:
      reward, _ = self.get_rewards(sim=sim)
      obs = []
      if get_obs:
        obs = self.get_obs()

      return obs
    else:
      return 0

  def step(self, action, get_obs=True):
    info = {}
    self.passes.append(action)
    reward, done = self.get_rewards()
    obs = []
    if get_obs:
      obs = self.get_obs()
    return (obs, reward, done, info)
    
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

