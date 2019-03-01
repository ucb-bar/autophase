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
import math
# Init: 8 progs, 8 envs
# Reset: reset pgm_count
# Step: (prog) % #prog
class HLSMultiEnv(gym.Env):
  def __init__(self, env_config):
    self.action_space = Discrete(45)
    self.observation_space= Box(0.0,1.0,shape=(56,),dtype = np.float32)
    self.num_pgms = 6
    self.envs = []
    self.idx = np.random.randint(self.num_pgms)

    from gym_hls.envs.chstone_bm import get_chstone, get_orig6
    bms = get_orig6()
    for i, bm in enumerate(bms):
      pgm, path = bm
      env_conf = {}
      env_conf['pgm'] = pgm
      env_conf['pgm_dir'] = path
      env_conf['run_dir'] = 'run_'+str(i)
      env_conf['normalize']=env_config.get('normalize',False)
      env_conf['log_obs_reward']=env_config.get('log_obs_reward',False)
      self.envs.append(HLSEnv(env_conf))

  def reset(self):
    self.idx = (self.idx + 1)  % self.num_pgms
    #print("idx -- ", self.idx)
    obs = self.envs[self.idx].reset()
    return obs

  def step(self, action):
    obs, reward, done, info = self.envs[self.idx].step(action)
    #print("action -- ", action," reward -- ", reward)
    #log_obs = [math.log(e+1) for e in obs]
    #log_reward = np.sign(reward) * math.log(abs(reward)+1)
    #print(log_obs)
    #print(log_reward)
    return obs, reward, done, info


  def render():
    self.envs[self.idx].render()
