from gym_hls.envs.hls_env import HLSEnv
def getbaseline(bm_name='chstone', num_pgms=None, clang_opt=False):
  import time
  if bm_name == "chstone": 
    from chstone_bm import get_chstone, get_others, get_all9
    bms = get_all9()
  elif bm_name == "random":
    from gym_hls.envs.random_bm import get_random
    bms = get_random(N=num_pgms)
  else:
    raise
  print(len(bms))
  
  fout = open("report_baseline"+".txt", "w")
  fout.write("Benchmark|-O0|-O3|-O0 Runtime(s)|-O3 Runtime(s)\n")

  for bm in bms:
    if bm_name == "chstone": 
      pgm, path = bm
      env_conf = {}
      env_conf['pgm'] = pgm
      env_conf['pgm_dir'] = path
      env_conf['run_dir'] = 'run_'+pgm.replace(".c","")
    elif bm_name == "random":
      pgm, files = bm
      env_conf = {}
      env_conf['pgm'] = pgm
      env_conf['pgm_files'] = files
      env_conf['run_dir'] = 'run_'+pgm.replace(".c","")
    else:
      raise
    
    env_conf['delete_run_dir'] = False
    env = HLSEnv(env_conf)

    try:
      begin = time.time()
      o0_cycle = - env.get_Ox_rewards(level=0, clang_opt=clang_opt)
      end = time.time()
      o0_compile_time = end - begin

      begin = time.time()
      o3_cycle = - env.get_Ox_rewards(level=3, clang_opt=clang_opt)
      end = time.time()
      o3_compile_time = end - begin
      env.__del__()
      fout.write("{}|{}|{}|{}|{}\n".format(pgm, o0_cycle, o3_cycle, o0_compile_time, o3_compile_time))
      print("{}|{}|{}|{}|{}\n".format(pgm, o0_cycle, o3_cycle, o0_compile_time, o3_compile_time))
    finally:
      env.__del__()
      del env
      import os
      os.system("rm -rf run_*")

  fout.close()

#getbaseline('random')
#getbaseline('chstone')
