#!/usr/bin/env python
#
# Optimize blocksize of apps/mmm_block.cpp
#
# This is an extremely simplified version meant only for tutorials
#
from __future__ import print_function
#import adddeps  # fix sys.path

import opentuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter, BooleanParameter, PermutationParameter
from opentuner import MeasurementInterface
from opentuner import Result
from gym_hls.envs.hls_env import HLSEnv as Env
import time

import pickle

NUM_PASSES = 16
class GccFlagsTuner(MeasurementInterface):
  def __init__(self, envs, *pargs, **kwargs):
    super(GccFlagsTuner, self).__init__(program_name='hls', *pargs,
                                        **kwargs)
    self.sample_size = 0
    #self.configs = configs
    self.envs = envs 
    self.min_cycles = 10000000
    self.log_file = open("opentuner_{}.log".format('hls'),"w")
    self.passes = []

  @classmethod
  def main(cls, envs, args, *pargs, **kwargs):
    from opentuner.tuningrunmain import TuningRunMain
    main = TuningRunMain(cls(envs, args, *pargs, **kwargs), args).main()
    return (self.min_cycles, self.passes, self.sample_size, main)

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(PermutationParameter('hls_order', list(range(NUM_PASSES)))
      )
    for i in range(NUM_PASSES):
      manipulator.add_parameter(BooleanParameter('hls_pass_{}'.format(i)))
    return manipulator

  def call_program(self, cmd, limit=None, memory_limit=None, **kwargs):
    rews = [env.get_cycles(cmd) for env in self.envs]
    time = sum(rews)
    #time = self.env.get_cycles(cmd)
    if time < self.min_cycles:
      self.min_cycles = time
      self.passes = cmd

    self.log_file.write("{}|{}|{}|{}|{}\n".format('hls', time, self.min_cycles, self.sample_size, cmd))
    return {'time': time,
            'timeout': False,
            'returncode': 0}

  def compile(self, cfg, id):
    """
    Compile a given configuration in parallel
    """
    order = cfg['hls_order']
    passes = []
    #print(order)
    for i in range(NUM_PASSES):
      switches = cfg['hls_pass_{}'.format(i)]
      #print(switches)
    for index in order:
      if cfg['hls_pass_{}'.format(index)]:  
        passes.append(index)
    self.sample_size += 1
    print(passes)
    print(self.sample_size)
    return self.call_program(passes)
  
  def run_precompiled(self, desired_result, input, limit, compile_result, id):
    """
    Run a compile_result from compile() sequentially and return performance
    """
    assert compile_result['returncode'] == 0

    try:    
        run_result = self.call_program('./tmp{0}.bin'.format(id))
        assert run_result['returncode'] == 0
    finally:
        self.call_program('rm ./tmp{0}.bin'.format(id))

    return Result(time=run_result['time'])

  def compile_and_run(self, desired_result, input, limit):
    """
    Compile and run a given configuration then
    return performance
    """
    cfg = desired_result.configuration.data
    run_result = self.compile(cfg, 0)
    #return self.run_precompiled(desired_result, input, limit, compile_result, 0)
    return Result(time=run_result['time'])


  def save_final_config(self, configuration):
    """called at the end of tuning"""
    print("Optimal block size written to json file:", configuration.data)
    #self.manipulator().save_to_file(configuration.data,
    #                                '{}.json'.format(self.configs['pgm'].replace(".c","")))
    print(configuration.data)
    #self.log_file.write("{}\n".format(configuration.data))

def ot_test_pgm_group(bms, N=1, test_len=[16]):
  argparser = opentuner.default_argparser()
  with open("ot_test_pgm_group.txt", "w") as fout: 
    fout.write("Benchmark | Cycle Counts | Algorithm Runtime (s)| Sample Sizes | Passes \n")

    for length in test_len:
      envs = []
      i = 0
      for pgm, path in bms:
        env_config = {
          'pgm':pgm,
          'pgm_files':path,
          'run_dir':'run_'+pgm.replace(".c",""),
          'normalize':False,
          'orig_and_normalize':False,
          'log_obs_reward':False,
          'verbose':False,
          'shrink':True,
          }

        envs.append(Env(env_config))
        i = i+1
      begin = time.time()
      cycles, passes, sample_size,_ = GccFlagsTuner.main(envs, argparser.parse_args())
      end = time.time()
      print("Best individuals are: {}".format(passes))
      print("Cycles: {}".format(timings[0]))
      compile_time = end - begin
      print("Compile Time: %d"%(int(compile_time)))
      fout.write("{}|{}|{}|{}|{}\n".format("test_pgm_group", cycles, compile_time, sample_size, passes))



def ot_test_pgm(bms, test_len=[16]):
  argparser = opentuner.default_argparser()
  #argparser.add_argument('source', help='source file to compile')
  # from gym_hls.envs.chstone_bm import get_chstone, get_others, get_all9
  #bms = get_all9()
  #bms = bms[0:1]
  # for i, bm in enumerate(bms):
  #   pgm, path = bm
  #   env_configs = {}
  #   env_configs['pgm'] = pgm
  #   env_configs['pgm_dir'] = path
  #   env_configs['run_dir'] = 'run_'+str(i)
  #   env_configs['verbose'] = True
  #   env_configs['log_results'] = True

  for i, bm in enumerate(bms):
    pgm, files= bm
    env_configs = {}
    env_configs['pgm'] = pgm 
    env_configs['pgm_files'] = files
    env_configs['run_dir'] = 'run_'+pgm.replace(".c","")
    #env_configs['feature_type'] = 'act_hist'
    env_configs['verbose'] = True
    env_configs['log_results'] = True

    print("Tune for {}".format(pgm))
    GccFlagsTuner.main(env_configs, argparser.parse_args())


if __name__ == '__main__':
    from gym_hls.envs.random_bm import get_random
    num_pgms = 100
    bms = get_random(N=num_pgms)
    #print(len(bms))
    ot_test_pgm_group(bms, [16])

