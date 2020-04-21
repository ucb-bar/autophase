from gym_hls.envs import getcycle
from gym_hls.envs import getfeatures
import os
import datetime
import glob
import shutil
import numpy as np
import gym
from gym.spaces import Discrete, Box, Tuple
import sys
from IPython import embed
import math
import pickle

class HLSEnv(gym.Env):
  """
    Attributes:
      pass_len (int)
      feat_len (int)
      eff_pass_indices (list)
          eff_feat_indices (list)
      binary_obs (bool)
          norm_obs (bool)
      orig_norm_obs (list)
          feature_type (str)
          bandit (bool)
      action_pgm (bool)
          action_meaning (list)
      reset_actions (list)
          max_episode_steps (int)

         action_space (Tuple)
      observation_space (Box)
          prev_cycles (int)
      O0_cycles (int)
      prev_obs (list)
          min_cycles (int) 
      verbose (bool)  
      log_obs_reward (bool)
          delete_run_dir (bool)
      init_with_passes (bool)
      log_results (bool)
          run_dir (str)
          log_file (file)
          pre_passes_str (str)
      pre_passes (list)
      passes (list)
          best_passes (list)
      pgm_name (str)
          bc (str)
          original_obs(list)

  """

  def __init__(self, env_config):
    """
    Args:
      Env_config (dict):  env_config is a dictionary that used to specify different settings, such as using the program’s features or the previous histogram of 
      actions as input of the Rl agent (for the observation)
  
    """

    self.pass_len = 45 # pass_len (int): number of passes for the program 
    self.feat_len = 56 # feat_len (int): number of features for the program 

    self.shrink = env_config.get('shrink', False) 
    if self.shrink:
      # eff_pass_indices (list): list of integers that represent the indices to be used for efficient pass ordering
      self.eff_pass_indices = [1,7,11,12,14,15,23,24,26,28,30,31,32,33,38,43 ]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44] 
      self.pass_len = len(self.eff_pass_indices) 
      # eff_feat_indices (list): list of integers that represent the indices to be used for efficient/important feature detection
      self.eff_feat_indices = [5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 31, 32, 33, 34, 36, 37, 38, 40, 42, 46, 49, 52, 55] 
      self.feat_len = len(self.eff_feat_indices) 
    # binary_obs (bool): binary_obs is a boolean set to True if we want list of features’ element to be either 1 or 0, and set to False otherwise.
    self.binary_obs = env_config.get('binary_obs', False)
    # norm_obs (bool): norm_obs is a Boolean set to True if we want to normalize all the elements of the features’ observations list to 1, and set to False otherwise 
    self.norm_obs = env_config.get('normalize', False) 
    # orig_norm_obs (list): orig_norm_obs is a Boolean set to True if we want the features’ observation list to contains both the original features values and the normalized ones. It is set to False otherwise.
    self.orig_norm_obs = env_config.get('orig_and_normalize', False)
    # feature_type (str): feature_type is a string that we set to determine what should the Rl use as observation features (as input). For example, we can set it to “pgm” if we want the Rl agent to only use the program’s features as observation, 
    # or “act_hist” if we want the Rl agent to use the histogram of previously applied passes as  the observation input.
    self.feature_type = env_config.get('feature_type', 'pgm') # pmg or act_hist
    self.act_hist = [0] * self.pass_len
    # bandit (bool): bandit is a Boolean that should be set to True if we want our action space to be a Tuple of 12 elements, where each element can be any of the pass. It is set to False Otherwise.
    self.bandit = self.feature_type == 'bandit'
    # action_pgm (bool): action_pgm is a Boolean that should be set to True if we our action space to be a Tuple which size is equal to the number of passes we have, and each element can be set to 0, 1 or 2. 
    # (This is described in the research paper as Conﬁguration2: Multiple-Action). It should be set to False otherwise.
    self.action_pgm = self.feature_type == 'act_pgm'
    # action_meaning (list): is a list of 3 integers (-1, 0, 1) that represent the value each pass can have in the action space for configuration 2 (in action_pgm), since we use the spaces.Discrete() fct these values are represented as 0, 1 or 2
    self.action_meaning = [-1,0,1]
    # reset_actions (list): is a list that of integers which size is equal to the number of passes, and all the elements are equal to the number of passes flour 2. This is also done to set for configuration 2. 
    self.reset_actions = [int(self.pass_len // 2)] * self.pass_len
    # max_episode_steps (int): represents how many times we will train the Rl agent (by feeding it obersvations, and count of cycles and making it produce new Ir for the program) 
    self.max_episode_steps=45
    if self.action_pgm:
        # action_space (Tuple): action_space is a Tuple (of integers) or merely an integer that defines the range of possible states the Rl agent might consider and the possible actions that the agent can take
        self.action_space=Tuple([Discrete(len(self.action_meaning))]*self.pass_len)
    elif self.bandit:
        self.action_space = Tuple([Discrete(self.pass_len)]*12)
    else:
        self.action_space = Discrete(self.pass_len)

    if self.feature_type == 'pgm':
        if self.orig_norm_obs:
          # observation_space (Box): observation_space is a space.Box() that represent the dimensions of the observations we are feeding the Rl agent as input for training. The first 2 parameters of the Box function show the 
          # bounds of the dimensions and the third parameter (shape) represent the number of dimensions of the parameter observation_space, finally the last parameter represent the type of each element in the Box function.
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

    self.prev_cycles = 10000000 # prev_cycles (int): prev_cycles is the number of cycle counts that it took to run the previous synthesized circuit. We use it to determine the reward of the Rl agent.
    self.O0_cycles = 10000000 # O0_cycles (int): O0_cycles is the number of cycle counts that it took to run the synthesized  circuit using the -O0 optimization level
    self.prev_obs = None # prev_obs (list): prev_obs is an array of numbers that represent the previous observation features used in the previous episode
    self.min_cycles = 10000000 # min_cycles (int): min_cycle keeps track of the minimum number of cycle counts recorded from running the synthesized cicuit simulation of the program. 
    self.verbose = env_config.get('verbose',False) # verbose (bool): verbose is a Boolean that should be set to True when we want to print the format or information about the results. It should be set to False otherwise.
    self.log_obs_reward = env_config.get('log_obs_reward',False) # log_obs_reward (bool): log_obs_reward is a Boolean that should be set to True if we want to normalize our observation features (by taking the log of every element from the observation list) in order for the RL agen tot work on new programs. It should be set to False otherwise.

    pgm = env_config['pgm']
    pgm_dir = env_config.get('pgm_dir', None)
    pgm_files = env_config.get('pgm_files', None)
    run_dir = env_config.get('run_dir', None)
    self.delete_run_dir = env_config.get('delete_run_dir', True) # delete_run_dir (bool):  delete_run_dir is a Boolean that should be set to True if we want to close the log_file. It should be set to False otherwise.
    self.init_with_passes = env_config.get('init_with_passes', False) # init_with_passes (bool): init_with_passes is a Boolean that should be set to True when we want to reset the episodes but want to start with specific passes. Basically, it reinitialize the Rl training with specific predetermined passes. It should be set to False otherwise.
    self.log_results = env_config.get('log_results', False) # log_results (bool): log_results is a Boolean that should be set to True when we want to create and write on the log_file. It should be set to False otherwise.

    if run_dir: # run_dir (str): run_diris the path of the running directory
        self.run_dir = run_dir+'_p'+str(os.getpid())
    else:
        currentDT = datetime.datetime.now()
        self.run_dir ="run-"+currentDT.strftime("%Y-%m-%d-%H-%M-%S-%f")+'_p'+str(os.getpid())

    if self.log_results:
        self.log_file = open(self.run_dir+".log","w") # log_file (file): log_file is a file we create and write on it information about the each episode (step) of the training. (record prev obs, reward, previous cycles ..)

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
    # pre_passes_str (str): pre_passes_str is a string that contains specific passes that we want to use when we reinitialize the training (when we use reset)
    self.pre_passes_str= "-prune-eh -functionattrs -ipsccp -globalopt -mem2reg -deadargelim -sroa -early-cse -loweratomic -instcombine -loop-simplify"
    # pre_passes (list): pre_passes is a list of integer that contains the indices of the passes written in pre_passes_str.
    self.pre_passes = getcycle.passes2indice(self.pre_passes_str) 
    self.passes = [] # passes (list): passes is a list that contains the passes used for the Rl training
    self.best_passes = [] # best_passes (list): best_passes is a list that contains the best passes recorded. (we update the list when the recoded time of cycle count is less than min_cycles)
    self.pgm = pgm # pgm_name (str): pgm_name is the file name of the program we are optimizing (which is written in C programming language)
    self.pgm_name = pgm.replace('.c','') # bc (str): bs is the file name of the program we are optimizing after being compiled to IR (hardware-independent intermediate representation)
    self.bc = self.pgm_name + '.prelto.2.bc'
    self.original_obs = [] # original_obs(list): original_obs is a list that contains the original values of the observatyions features.

  def __del__(self):

    """
    This function closes the log_file (which is a file we use to record information about each episode) when delete_run_dir and log_results are True.
    Also deletes the entire directory tree of run_dir (the running directory) if run_dir is an existing directory.
  
    """

    if self.delete_run_dir:
        if self.log_results:
            self.log_file.close()
    if os.path.isdir(self.run_dir):
        shutil.rmtree(self.run_dir)

  def get_Ox_rewards(self, level=3, sim=False, clang_opt=False):
    """
    Examples :
      >>> print(get_0x_rewards(self, level=3, clang_opt=False, sim=False))
      -45

    Args:
      level (int): This is an integer that represents different groups of optimizations implemented in the compiler. 
        Each optimization level is hand-picked by the compiler-designer to benefit specific benchmarks. Defaults to 3.
      sim (bool): sim is a Boolean that should be set to True if we want the subprocessor to run the “make clean p v -s” command, 
        and we should set it to False if we want the subprocessor to run the “make clean accelerationCycle -s” command instead. Defaults to False.
      clang_opt (bool): clang_opt is a Boolean that should be set to True if we want to use the clang option when running the HLS, and should be set to False otherwise.

    Returns:
      Returns the negative number of cycle counts it took to run the synthesized circuit made by using the passes set in the 0x optimization. Which represents for the RL agent the reward.
  
    """

    from gym_hls.envs.getox import getOxCycles
    cycle = getOxCycles(self.pgm_name, self.run_dir, level=level, clang_opt=clang_opt, sim=sim)
    return -cycle

  def print_info(self,message, end = '\n'):
    """
      This function is used to print information the episodes of the RL agent.

    Args:
      message (str): message is a string that will contain information about the episode of the Rl agent that we want to print on our terminal
      end (str): end is a string  that prints a new line. 
  
    """

    sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

  def get_cycles(self, passes, sim=False):
    """
    Examples :
      >>>print(get_cycles(self, [“-correlated-propagation”, “-scalarrepl”, “-lowerinvoke”])) 
      (55, True)

    Args:
        passes (list): passes is a list that contains the passes used for the Rl training
        sim (bool): sim (bool, optional): sim should be True if you want the arguments used to launch the process to be “make clean p v -s”, or sim should 
          be False if you want the argument used to launch the process to be "make clean accelerationCycle -s". Defaults to False

    Returns:
      Returns a tuple where the first element is an integer that represents the number of cycle counts it took to run the synthesized circuit 
      (the second element doesn’t matter).
    """

    if self.shrink:
          actual_passes = [self.eff_pass_indices[index] for index in passes]
    else:
        actual_passes =  passes

    cycle, _ = getcycle.getHWCycles(self.pgm_name, actual_passes, self.run_dir, sim=sim)
    return cycle

  def get_rewards(self, diff=True, sim=False):
    """
    Examples :
      >>>print(get_cycles(self)) 
      -55

    Args:
      diff (bool): diff is a boolean that is set to True if we want the reward to be the difference of previous cycle count and the current cycle count. 
        Otherwise, if diff is False, the reward is equal to – the current cycle count.
      sim (bool, optional): sim should be True if you want the arguments used to launch the process to be “make clean p v -s”, or sim should be False if you want the 
        argument used to launch the process to be "make clean accelerationCycle -s". Defaults to False

    Returns:
      Returns an integer that represents the reward for the RL agent (it shows the improvement of the circuit), and we get it either by calculating the difference 
      between previous cycle count and the current cycle count or the negative value of the current cycle count.

    """

    if self.shrink:
        actual_passes = [self.eff_pass_indices[index] for index in self.passes]
    else:
        actual_passes =  self.passes
    cycle, done = getcycle.getHWCycles(self.pgm_name, actual_passes, self.run_dir, sim=sim)
    if cycle == 10000000:
        cycle = 2 * self.O0_cycles

   # print("pass: {}".format(self.passes))
   # print("prev_cycles: {}".format(self.prev_cycles))
    if(self.verbose):
        self.print_info("passes: {}".format(actual_passes))
        self.print_info("program: {} -- ".format(self.pgm_name)+" cycle: {}  -- prev_cycles: {}".format(cycle, self.prev_cycles))
        try:
            cyc_dict = pickle.load(open('cycles_chstone.pkl','rb'))
        except:
            cyc_dict = {}
        if self.pgm_name in cyc_dict:
            if cyc_dict[self.pgm_name]['cycle']>cycle:
                cyc_dict[self.pgm_name]['cycle'] = cycle
                cyc_dict[self.pgm_name]['passes'] = self.passes
        else:
            cyc_dict[self.pgm_name] = {}
            cyc_dict[self.pgm_name]['cycle'] = cycle
            cyc_dict[self.pgm_name]['passes'] = self.passes
        output = open('cycles_chstone.pkl', 'wb')
        pickle.dump(cyc_dict, output)
        output.close()

    if (cycle < self.min_cycles):
      self.min_cycles = cycle
      self.best_passes = actual_passes
    if (diff):
        rew = self.prev_cycles - cycle
        self.prev_cycles = cycle
    else:
      rew = -cycle
   # print("rew: {}".format(rew))
    return rew, done

  def get_obs(self,get_normalizer=False):
    """
    Examples :
    >>>print(get_obs())
    [1, 0, 0, 0, 1]

    Args:
        get_normalizer (bool): get_normalizer is a boolean that should be set to True if we want to get a normalizer value that is used to normalize the list of 
        observation features. Defaults to False.

    Returns:
      Returns a list or a tuple that contains the list of the observation features that we need to feed as input to the RL agent.
  
    """

    feats = getfeatures.run_stats(self.bc, self.run_dir)
    normalizer=feats[-5] + 1
    if self.shrink:
        actual_feats = [feats[index] for index in self.eff_feat_indices]
    else:
        actual_feats = feats

    if self.binary_obs:
        actual_feats = [1 if feat > 0 else 0 for feat in actual_feats]
    if not get_normalizer:
        return actual_feats
    else:
        return actual_feats,normalizer
    return actual_feats

  # reset() resets passes to []
  # reset(init=[1,2,3]) resets passes to [1,2,3]
  def reset(self, init=None, get_obs=True, get_rew=False, ret=True, sim=False):
    """
    Examples :
    >>>print(reset())
    [0, 0, 0, 0]

    Args:
      init (list, optional): init is a list of integer that is equal to (set to) the new passes list. Defaults to None.
      get_obs (bool, optional): get_obs is a Boolean that is set to True when we decide to get the list of observation features after we reset. 
        It should be set to False otherwise. Defaults to True.
      get_rew (bool, optional): get_rew is a Boolean that is set to True when we decide to get the reward after we reset. It should be set to False otherwise. 
        Defaults to False.
      ret (bool, optional): ret is a Boolean that is set to True when we decide to get the reward or the list of observation features after we reset. 
        It should be set to False otherwise. Defaults to True.
      sim (bool, optional): sim should be True if you want the arguments used to launch the process to be “make clean p v -s”, or sim should be False if you want 
        the argument used to launch the process to be "make clean accelerationCycle -s". Defaults to False. Defaults to False.

    Returns:
      Returns an integer for the reward or a list for the observation features, or a tuple of both an integer and a list for the reward and the observation features, or zero if ret if False.
    """

    #self.min_cycles = 10000000

    self.passes = []
    if self.feature_type == 'act_pgm':
        self.passes = self.reset_actions
    if self.init_with_passes:
      self.passes.extend(self.pre_passes)

    if init:
      self.passes.extend(init)

    self.prev_cycles = self.get_cycles(self.passes)
    self.O0_cycles = self.prev_cycles
    if(self.verbose):
        self.print_info("program: {} -- ".format(self.pgm_name)+" reset cycles: {}".format(self.prev_cycles))
    if ret:
      if get_rew:
        reward, _ = self.get_rewards(sim=sim)
      obs = []
      if get_obs:
        if self.feature_type == 'pgm':
          obs = self.get_obs()

          if self.norm_obs or self.orig_norm_obs:
            self.original_obs = [1.0*(x+1) for x in obs]
            relative_obs = len(obs)*[1]
            if self.norm_obs:
              obs = relative_obs
            elif self.orig_norm_obs:
              obs = list(self.original_obs)
              obs.extend(relative_obs)
            else:
              raise
          if self.log_obs_reward:
            if  (self.norm_obs or self.orig_norm_obs):
                log_obs = [math.log(e) for e in obs]
            else:
                log_obs = [math.log(e+1) for e in obs]
            obs = log_obs

        elif self.feature_type == 'act_hist' or self.feature_type == "act_hist_sparse":
          self.act_hist = [0] * self.pass_len
          obs = self.act_hist
        elif self.feature_type == 'act_pgm':
          obs = self.reset_actions+self.get_obs()
        elif self.feature_type == 'hist_pgm':
          self.act_hist = [0] * self.pass_len
          obs,normalizer = self.get_obs(get_normalizer=True)
          obs = self.act_hist + [1.0*f/normalizer for f in obs]
        elif self.bandit:
          obs = [1] * 12
        else:
          raise

        obs = np.array(obs)
        if self.log_results:
          self.prev_obs = obs

      if get_rew and not get_obs:
        return reward
      if get_obs and not get_rew:
        return obs
      if get_obs and get_rew:
        return (obs, reward)
    else:
      return 0

  def step(self, action, get_obs=True):
    """
    Examples :
      >>>print(step(action))
      ([1.54, 0.76, 0.99], 34, True, {})

    Args:
      action (list): action is a list of the passes that the RL decide to apply as the next move after having analyzed its input values (observation features list and reward from the cycle count).
      get_obs (bool, optional): get_obs is a Boolean that is set to True when we decide to get the list of observation features during each step. It should be set to False otherwise. Defaults to True.

    Returns:
      Returns a tuple of observation features list, reward from cycle count, the Boolean done from get_reward, and info (the dictionary initialized at the beginning of the function).
  
    """

    info = {}
    if self.bandit:
        self.passes = action
    elif self.feature_type =='act_pgm':
        for i in range(self.pass_len):
            action = np.array(action).flatten()
            self.passes[i] = (self.passes[i]+self.action_meaning[action[i]])%self.pass_len
            if self.passes[i] > self.pass_len - 1:
                self.passes[i] = self.pass_len - 1
            if self.passes[i] < 0:
                self.passes[i] = 0
    else:
        self.passes.append(action)

    if self.feature_type == "act_hist_sparse" and len(self.passes) <  self.max_episode_steps:
      reward = 0
      done = False
    else:
      reward, done = self.get_rewards()

    obs = []
    if(self.verbose):
        self.print_info("program: {} --".format(self.pgm_name) + "passes: {}".format(self.passes))
        self.print_info("reward: {} -- done: {}".format(reward, done))
        self.print_info("min_cycles: {} -- best_passes: {}".format(self.min_cycles, self.best_passes))
        self.print_info("act_hist: {}".format(self.act_hist))

    if get_obs:

      if self.feature_type == 'pgm':
        obs = self.get_obs()
        if self.norm_obs or self.orig_norm_obs:
          relative_obs =  [1.0*(x+1)/y for x, y in zip(obs, self.original_obs)]
          if self.norm_obs:
            obs = relative_obs
          elif self.orig_norm_obs:
            obs =  [e+1 for e in obs]
            obs.extend(relative_obs)
          else:
            raise

        if self.log_obs_reward:
          if self.norm_obs or self.orig_norm_obs:
            obs = [math.log(e) for e in obs]
          else:
            obs = [math.log(e+1) for e in obs]
          reward = np.sign(reward) * math.log(abs(reward)+1)

      elif self.feature_type == 'act_hist' or self.feature_type == "act_hist_sparse":
        self.act_hist[action] += 1
        obs = self.act_hist
      elif self.feature_type == 'act_pgm':
        obs = self.passes + self.get_obs()
      elif self.feature_type == 'hist_pgm':
        self.act_hist[action] += 1
        obs,normalizer = self.get_obs(get_normalizer=True)
        obs = self.act_hist + [1.0*f/normalizer for f in obs]
        reward = np.sign(reward) * math.log(abs(reward)+1)
        if reward<0 and action in self.passes[:-1]:
            reward = reward-10
      elif self.bandit:
        obs = self.passes

    obs = np.array(obs)
    if self.log_results:
      if self.feature_type == "act_hist_sparse" and (len(self.passes) == self.max_episode_steps):
        #self.log_file.write("{}, {}, {}, {}, {}\n".format(self.prev_obs, action, reward, self.prev_cycles, self.min_cycles))
        print("{}|{}|{}|{}|{}|{}|{}\n".format(self.prev_obs, action, reward, self.prev_cycles, self.min_cycles, self.passes, self.best_passes))
        self.log_file.write("{}|{}|{}|{}|{}|{}|{}\n".format(self.prev_obs, action, reward, self.prev_cycles, self.min_cycles, self.passes, self.best_passes))
      else:
        self.log_file.write("{}|{}|{}|{}|{}|{}|{}\n".format(self.prev_obs, action, reward, self.prev_cycles, self.min_cycles, self.passes, self.best_passes))
      self.log_file.flush()

    self.prev_obs = obs
    return (obs, reward, done, info)

  def multi_steps(self, actions):
    """
    Args:
      actions (list): actions is a list of the passes that the RL decide to apply as the next move after 
        having analyzed its input values (observation features list and reward from the cycle count).

    Returns:
      Returns a tuple of the new observation features list and the new reward after the RL agent applied
       its action (after applying optimization passes).
  
    """

    rew = self.get_rewards()
    self.passes.extend(actions)
    obs = self.get_obs()
    if self.norm_obs:
      relative_obs =  [1.0*(x+1)/y for x, y in zip(obs, self.original_obs)]
      relative_obs.extend(obs)

    return (self.get_obs(), self.get_rewards())

  def render():
    """
    This function prints information about the passes and the previous cycles count.  
    """

    print("pass: {}".format(self.passes))
    print("prev_cycles: {}".format(self.prev_cycles))


def getOx():
  """
  This function gets a variety of benchmarks from the chstone suite benchmark and the get_others suite then
   create a text file name report_O3 and write in it the cycle count, algorithm runtime, and the passes of each program.  
  """

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
  """
  This function returns the geomean of different programs that run on the same environment 
  and start with the same test passes (as initial passes for the Rl agent).
  """

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

