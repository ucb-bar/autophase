from gym_hls.envs import getcycle
from gym_hls.envs import getfeatures
import os
import datetime
import glob
import shutil
import numpy as np
import gym
from gym.spaces import Tuple, Discrete, Box
from gym_hls.envs.hls_env import HLSEnv
import math
# Init: 8 progs, 8 envs
# Reset: reset pgm_count
# Step: (prog) % #prog
class HLSMultiEnv(gym.Env):
  def __init__(self, env_config):

    self.pass_len = 45
    self.feat_len = 56

    self.shrink = env_config.get('shrink', False)
    if self.shrink:
      # TODO FIX Duplication
      self.eff_pass_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44] 
      self.pass_len = len(self.eff_pass_indices) 
      self.eff_feat_indices = [5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 31, 32, 33, 34, 36, 37, 38, 40, 42, 46, 49, 52, 55]
      self.feat_len = len(self.eff_feat_indices) 

    bm_name = env_config.get('bm_name', 'chstone')
    self.num_pgms = env_config.get('num_pgms', 6)
    self.norm_obs = env_config.get('normalize', False)
    self.orig_norm_obs = env_config.get('orig_and_normalize', False)
    self.feature_type = env_config.get('feature_type','pgm')
    self.action_pgm = self.feature_type == 'act_pgm'
    self.bandit = self.feature_type == 'bandit'

    if self.action_pgm:
        self.action_space=Tuple([Discrete(len(self.action_meaning))]*self.pass_len)
    elif self.bandit:
        self.action_space = Tuple([Discrete(self.pass_len)]*12)
    else:
        self.action_space = Discrete(self.pass_len)

    if self.feature_type == 'pgm':
      if self.orig_norm_obs:
        self.observation_space = Box(0.0,1.0,shape=(self.feat_len*2,),dtype = np.float32)
      else:
        self.observation_space = Box(0.0,1000000,shape=(self.feat_len,),dtype = np.int32)
    elif self.feature_type == 'act_hist' or self.feature_type == "act_hist_sparse":
      self.observation_space = Box(0.0,45,shape=(self.pass_len,),dtype = np.int32)
    elif self.feature_type == 'act_pgm' or self.feature_type == 'hist_pgm':
      if self.orig_norm_obs:
        feat_len = self.feat_len * 2
      else:
        feat_len = self.feat_len
      self.observation_space = Box(0.0,1.0,shape=(self.pass_len+feat_len,),dtype = np.float32)
    elif self.bandit:
      self.observation_space = Box(0.0,1.0,shape=(12,),dtype = np.float32)

    else:
      raise

    self.envs = []
    self.idx = np.random.randint(self.num_pgms)

    env_conf = {}
    env_conf['feature_type'] = self.feature_type
    env_conf['verbose'] = env_config.get('verbose',False)
    env_conf['orig_and_normalize'] = self.orig_norm_obs
    env_conf['log_obs_reward']=env_config.get('log_obs_reward',False)
    env_conf['shrink'] = env_config.get('shrink', False)
    env_conf['log_results'] = env_config.get('log_results', False)

    if bm_name == "chstone":
      from gym_hls.envs.chstone_bm import get_chstone, get_all9
      bms = get_all9()
      for i, bm in enumerate(bms):
        pgm, path = bm
        env_conf['pgm'] = pgm
        env_conf['pgm_dir'] = path
        env_conf['run_dir'] = 'run_'+pgm.replace(".c","")
        env_conf['normalize'] = self.norm_obs
        self.envs.append(HLSEnv(env_conf))
    elif bm_name == "random":
      from gym_hls.envs.random_bm import get_random
      bms = get_random(N=self.num_pgms)
      for i, bm in enumerate(bms):
        pgm, files = bm
        env_conf['pgm'] = pgm
        env_conf['pgm_files'] = files
        env_conf['run_dir'] = 'run_'+pgm.replace(".c","")
        env_conf['normalize'] = self.norm_obs
        self.envs.append(HLSEnv(env_conf))
    else:
      raise

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

def test():
  env_config = {
    'normalize':False,
    'orig_and_normalize':False,
    'log_obs_reward':True,
    'verbose':True,
    'bm_name':'random',
    'num_pgms':10}
  env = HLSMultiEnv(env_config)
  env.reset()
  cycle = env.envs[0].get_Ox_rewards(level=1)
  print(cycle)
  print(env.step(1))


if __name__ == "__main__":
  test()

