from gym_hls.envs import getcycle
from gym_hls.envs import getfeatures
import os
import datetime
import glob
import shutil
import numpy as np
import gym
from gym.spaces import Discrete, Box
import sys
from IPython import embed
import math
import pickle
class HLSEnv(gym.Env):
  def __init__(self, env_config):

    self.norm_obs = env_config.get('normalize', False)
    self.orig_norm_obs = env_config.get('orig_and_normalize', False)

    self.action_space = Discrete(45)

    if self.orig_norm_obs:
      self.observation_space = Box(0.0,1.0,shape=(56*2,),dtype = np.float32)
    else:
      self.observation_space = Box(0.0,1.0,shape=(56,),dtype = np.float32)

    self.prev_cycles = 10000000
    self.verbose = env_config.get('verbose',False)
    self.log_obs_reward = env_config.get('log_obs_reward',False)

    pgm = env_config['pgm']
    pgm_dir = env_config.get('pgm_dir', None)
    pgm_files = env_config.get('pgm_files', None)
    run_dir = env_config.get('run_dir', None)
    self.delete_run_dir = env_config.get('delete_run_dir', False)
    self.init_with_passes = env_config.get('init_with_passes', False)

    if run_dir:
      self.run_dir = run_dir+'_p'+str(os.getpid())
    else:
      currentDT = datetime.datetime.now()
      self.run_dir ="run-"+currentDT.strftime("%Y-%m-%d-%H-%M-%S-%f")+'_p'+str(os.getpid())

    cwd = os.getcwd()
    self.run_dir = os.path.join(cwd, self.run_dir)
    print(self.run_dir)
    if os.path.isdir(self.run_dir):
      shutil.rmtree(self.run_dir, ignore_errors=True)
    if pgm_dir:
      shutil.copytree(pgm_dir, self.run_dir)
    if pgm_files:
      os.makedirs(self.run_dir)
      for f in pgm_files:
        shutil.copy(f, self.run_dir)

    self.pre_passes_str= "-prune-eh -functionattrs -ipsccp -globalopt -mem2reg -deadargelim -sroa -early-cse -loweratomic -instcombine -loop-simplify"
    self.pre_passes = getcycle.passes2indice(self.pre_passes_str)
    self.passes = []
    self.pgm = pgm
    self.pgm_name = pgm.replace('.c','')
    self.bc = self.pgm_name + '.prelto.2.bc'
    self.original_obs = []

  def __del__(self):
    if self.delete_run_dir:
      if os.path.isdir(self.run_dir):
        shutil.rmtree(self.run_dir)

  def get_O3_rewards(self, sim=False):
    from gym_hls.envs.geto3 import getO3Cycles
    cycle = getO3Cycles(self.pgm_name, self.run_dir, sim=sim)
    return -cycle

  def print_info(self,message, end = '\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)


  def get_rewards(self, diff=True, sim=False):
    cycle, done = getcycle.getHWCycles(self.pgm_name, self.passes, self.run_dir, sim=sim)
   # print("pass: {}".format(self.passes))
   # print("prev_cycles: {}".format(self.prev_cycles))
    if(self.verbose):
        self.print_info("program: {} -- ".format(self.pgm_name)+" cycle: {}".format(cycle))
        cyc_dict = {self.pgm_name: cycle}
        output = open('cycles.pkl', 'ab')
        pickle.dump(cyc_dict, output)
        output.close()

    if (diff):
      rew = self.prev_cycles - cycle
      self.prev_cycles = cycle
    else:
      rew = -cycle
   # print("rew: {}".format(rew))
    return rew, done

  def get_obs(self):
    feat = getfeatures.run_stats(self.bc, self.run_dir)
    return feat

  # reset() resets passes to []
  # reset(init=[1,2,3]) resets passes to [1,2,3]
  def reset(self, init=None,get_obs=True, ret=True, sim=False):
    self.prev_cycles, _ = getcycle.getHWCycles(self.pgm_name, self.passes, self.run_dir, sim=sim)
    self.passes = []
    if(self.verbose):
        self.print_info("program: {} -- ".format(self.pgm_name)+" reset cycles: {}".format(self.prev_cycles))
    if self.init_with_passes:
      self.passes.extend(self.pre_passes)

    if init:
      self.passes.extend(init)

    if ret:
      reward, _ = self.get_rewards(sim=sim)
      obs = []
      if get_obs:
        obs = self.get_obs()

      if self.norm_obs or self.orig_norm_obs:
        self.original_obs = [1.0*(x+1) for x in obs]
        relative_obs = len(obs)*[1]
        if self.norm_obs:
          obs = relative_obs
        elif self.orig_norm_obs:
          obs.extend(relative_obs)
        else:
          raise

      if self.log_obs_reward:
        log_obs = [math.log(e+1) for e in obs]
        return log_obs
      return obs
    else:
      return 0

  def step(self, action, get_obs=True):
    info = {}
    if(self.verbose):
        self.print_info("program: {} --".format(self.pgm_name) + " action: {}".format(action))
    self.passes.append(action)
    reward, done = self.get_rewards()
    obs = []
    if get_obs:
      obs = self.get_obs()

    if self.norm_obs or self.orig_norm_obs:
      relative_obs =  [1.0*(x+1)/y for x, y in zip(obs, self.original_obs)]
      if self.norm_obs:
        obs = relative_obs
      elif self.orig_norm_obs:
        obs.extend(relative_obs)
      else:
        raise

    if self.log_obs_reward:
        obs = [math.log(e+1) for e in obs]
        reward = np.sign(reward) * math.log(abs(reward)+1)

    return (obs, reward, done, info)

  def multi_steps(self, actions):
    rew = self.get_rewards()
    self.passes.extend(actions)
    obs = self.get_obs()
    if self.norm_obs:
      relative_obs =  [1.0*(x+1)/y for x, y in zip(obs, self.original_obs)]
      relative_obs.extend(obs)

    return (self.get_obs(), self.get_rewards())

  def render():
    print("pass: {}".format(self.passes))
    print("prev_cycles: {}".format(self.prev_cycles))


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

