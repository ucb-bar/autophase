import getcycle
import os
import getfeatures
import datetime
import glob
import shutil
import numpy as np
import gym
from gym.spaces import Discrete, Box

# Init: 8 progs, 8 envs 
# Reset: reset pgm_count  
# Step: (prog) % #prog
# 1. reward 
# 2. 
class HLSMultiEnv(gym.Env):
  def __init__(self, env_config):
    self.action_space = Discrete(45)
    self.observation_space= Box(0.0,1.0,shape=(56,),dtype = np.float32)

    self.num_pgms = 8 
    self.envs = []
    self.prev_cycles = [0] * 8  
    self.idx = np.random.randint(self.num_pgms) 

    from chstone_bm import get_chstone, get_others
    from env import Env
    bms = get_chstone(N=self.num_pgms)
    for i, bm in enumerate(bms):
      pgm, path = bm
      self.envs.append(Env(pgm, path, str(i)))
  
  def reset(self):
    self.idx = (self.idx + 1)  % self.num_pgms  
    obs, cycles = self.envs[self.idx].reset()
    self.prev_cycles[self.idx] = cycles
    
    return obs

  def step(self, action):
    obs, cycles, done = self.envs[self.idx].step(action)
    reward = self.prev_cycles[self.idx] - cycles
    self.prev_cycles[self.idx] = cycles
    info = {}
    return obs, reward, done, info


