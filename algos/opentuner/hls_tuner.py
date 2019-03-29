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
from gym_hls.envs.hls_env import HLSEnv

import pickle

class GccFlagsTuner(MeasurementInterface):
  def __init__(self, configs, *pargs, **kwargs):
    super(GccFlagsTuner, self).__init__(program_name=configs['pgm'], *pargs,
                                        **kwargs)
    self.sample_size = 0
    self.configs = configs
    self.env = HLSEnv(configs)
    self.min_cycles =10000000
    self.log_file = open("opentuner_{}.log".format(configs['pgm']),"w")

  @classmethod
  def main(cls, configs, args, *pargs, **kwargs):
    from opentuner.tuningrunmain import TuningRunMain
    return TuningRunMain(cls(configs, args, *pargs, **kwargs), args).main()

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(PermutationParameter('hls_order', list(range(45)))
      )
    for i in range(45):
      manipulator.add_parameter(BooleanParameter('hls_pass_{}'.format(i)))
    return manipulator

  def call_program(self, cmd, limit=None, memory_limit=None, **kwargs):
    time = self.env.get_cycles(cmd)
    if time < self.min_cycles:
      self.min_cycles = time

    self.log_file.write("{}|{}|{}|{}|{}\n".format(self.configs['pgm'], time, self.min_cycles, self.sample_size, cmd))
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
    for i in range(45):
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
  

if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  #argparser.add_argument('source', help='source file to compile')
  from gym_hls.envs.chstone_bm import get_chstone, get_others, get_all9
  bms = get_all9()
  #bms = bms[0:1]
  
  for i, bm in enumerate(bms):
    pgm, path = bm
    env_configs = {}
    env_configs['pgm'] = pgm
    env_configs['pgm_dir'] = path
    env_configs['run_dir'] = 'run_'+str(i)
    env_configs['verbose'] = True
    env_configs['log_results'] = True

    print("Tune for {}".format(pgm))
    GccFlagsTuner.main(env_configs, argparser.parse_args())

