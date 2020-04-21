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
    """
    Args:
      Env_config (dict):  env_config is a dictionary that used to specify different settings, such as using the program’s features or the previous histogram of 
      actions as input of the Rl agent (for the observation)
  
    """

    self.pass_len = 45  # pass_len (int): number of passes for the program 
    self.feat_len = 56  # feat_len (int): number of features for the program 

    bm_name = env_config.get('bm_name', 'chstone')
    self.num_pgms = env_config.get('num_pgms', 6) # num_pgm (int): number of different program the RL agent should run.

    # norm_obs (bool): norm_obs is a Boolean set to True if we want to normalize all the elements of the features’ observations list to 1, and set to False otherwise 
    self.norm_obs = env_config.get('normalize', False)
    # orig_norm_obs (list): orig_norm_obs is a Boolean set to True if we want the features’ observation list to contains both the original features values and the normalized ones. It is set to False otherwise.
    self.orig_norm_obs = env_config.get('orig_and_normalize', False)

    # feature_type (str): feature_type is a string that we set to determine what should the Rl use as observation features (as input). For example, we can set it to “pgm” if we want the Rl agent to only use the program’s features as observation, 
    # or “act_hist” if we want the Rl agent to use the histogram of previously applied passes as  the observation input.
    self.feature_type = env_config.get('feature_type','pgm')
    self.shrink = env_config.get('shrink', False)
    if self.shrink:
      # eff_pass_indices (list): list of integers that represent the indices to be used for efficient pass ordering
      self.eff_pass_indices = [1,7,11,12,14,15,23,24,26,28,30,31,32,33,38,43 ]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
      self.pass_len = len(self.eff_pass_indices)
      # eff_feat_indices (list): list of integers that represent the indices to be used for efficient/important feature detection
      self.eff_feat_indices = [5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 31, 32, 33, 34, 36, 37, 38, 40, 42, 46, 49, 52, 55]
      self.feat_len = len(self.eff_feat_indices)
      
    # action_space (Tuple): action_space is a Tuple (of integers) or merely an integer that defines the range of possible states the Rl agent might consider and the possible actions that the agent can take.
    self.action_space = Discrete(self.pass_len)

    # action_meaning (list): is a list of 3 integers (-1, 0, 1) that represent the value each pass can have in the action space for configuration 2 (in action_pgm), since we use the spaces.Discrete() fct these values are represented as 0, 1 or 2
    self.action_meaning = [-1,0,1]
    if self.orig_norm_obs:

        # observation_space (Box): observation_space is a space.Box() that represent the dimensions of the observations we are feeding the Rl agent as input for training. The first 2 parameters of the Box function show the 
        # bounds of the dimensions and the third parameter (shape) represent the number of dimensions of the parameter observation_space, finally the last parameter represent the type of each element in the Box function.
        self.observation_space = Box(0.0,1.0,shape=(self.feat_len*2,),dtype = np.float32)
    elif self.feature_type == 'act_pgm':
        self.observation_space = Box(0.0,1.0,shape=(self.pass_len+self.feat_len,),dtype = np.float32)
        self.action_space=Tuple([Discrete(len(self.action_meaning))]*self.pass_len)
    elif self.feature_type == 'hist_pgm':
        self.observation_space = Box(0.0,1.0,shape=(self.pass_len+self.feat_len,),dtype = np.float32)
    else:
      self.observation_space = Box(0.0,1.0,shape=(self.feat_len,),dtype = np.float32)

    self.envs = []
    self.idx = np.random.randint(self.num_pgms)  # idx (int): random number between 0 and the number of programs we have to run.

    if bm_name == "chstone":
      from gym_hls.envs.chstone_bm import get_chstone, get_all9
      bms = get_all9()
      for i, bm in enumerate(bms):
        pgm, path = bm
        env_conf = {}
        env_conf['feature_type'] = self.feature_type
        env_conf['pgm'] = pgm
        env_conf['pgm_dir'] = path
        env_conf['run_dir'] = 'run_'+pgm.replace(".c","")
        env_conf['normalize'] = self.norm_obs
        env_conf['verbose'] = env_config.get('verbose',False)
        env_conf['orig_and_normalize'] = self.orig_norm_obs
        env_conf['log_obs_reward']=env_config.get('log_obs_reward',False)
        env_conf['log_results'] = env_config.get('log_results',False)
        env_conf['shrink'] = self.shrink
        self.envs.append(HLSEnv(env_conf))

    elif bm_name == "random":
      from gym_hls.envs.random_bm import get_random
      bms = get_random(N=self.num_pgms)
      for i, bm in enumerate(bms):
        pgm, files = bm
        env_conf = {}
        env_conf['feature_type'] = self.feature_type
        env_conf['pgm'] = pgm
        env_conf['pgm_files'] = files
        env_conf['run_dir'] = 'run_'+pgm.replace(".c","")
        env_conf['normalize'] = self.norm_obs
        env_conf['verbose'] = env_config.get('verbose',False)
        env_conf['orig_and_normalize'] = self.orig_norm_obs
        env_conf['log_obs_reward']=env_config.get('log_obs_reward',False)
        env_conf['log_results'] = env_config.get('log_results',False)
        env_conf['shrink'] = self.shrink
        self.envs.append(HLSEnv(env_conf))
    else:
      raise

  def reset(self): 
    """
    This function calls itself recursiveley by resetting all the programs simultaneously.
    """
    self.idx = (self.idx + 1)  % self.num_pgms
    #print("idx -- ", self.idx)
    obs = self.envs[self.idx].reset()
    return obs

  def step(self, action):
    """
    Examples :
      >>>print(step(action))
      ([1.54, 0.76, 0.99], 34, True, {})

    Args:
      action (list): action is a list of the passes that the RL decide to apply as the next move after having analyzed its input values (observation features list and reward from the cycle count).

    Returns:
      Returns a tuple of observation features list, reward from cycle count, the Boolean done from get_reward, and info (the dictionary) for the selectred program.

    """

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

