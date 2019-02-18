from gym_hls.envs import getcycle
from gym_hls.envs import getfeatures
import os
import datetime
import glob
import shutil
import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym_hls.envs.hls_env import HLSEnv

# Init: 8 progs, 8 envs 
# Reset: reset pgm_count 
# Step: (prog) % #prog
class HLSMultiEnv(gym.Env):
  def __init__(self, env_configs):
    self.action_space = Discrete(45)
    self.observation_space= Box(0.0,1.0,shape=(56,),dtype = np.float32)

    self.num_pgms = 8 
    self.envs = []
    self.idx = np.random.randint(self.num_pgms) 

    from gym_hls.envs.chstone_bm import get_chstone, get_others
    bms = get_chstone(N=self.num_pgms)
    for i, bm in enumerate(bms):
      pgm, path = bm
      env_configs = {}
      env_configs['pgm'] = pgm
      env_configs['pgm_dir'] = path
      env_configs['run_dir'] = 'run_'+str(i)
      self.envs.append(HLSEnv(env_configs))

  def reset(self):
    self.idx = (self.idx + 1)  % self.num_pgms  
    obs, rew = self.envs[self.idx].reset()
    return obs

  def step(self, action):
    obs, reward, done, info = self.envs[self.idx].step(action)
    return obs, reward, done, info


