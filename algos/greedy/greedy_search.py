import time
from gym_hls.envs import getcycle
from gym_hls.envs.hls_env import HLSEnv as Env
from multiprocessing.pool import ThreadPool
import sys, os
import pickle
import numpy as np

def geo_mean(iterable):
    a = np.array(iterable).astype(float)
    prod = a.prod()
    prod = -prod if prod < 0 else prod
    return prod**(1.0/len(a))

   
NUM_PASSES = 45


def get_lookup_rank(envs): 
  cycles = np.zeros(NUM_PASSES)
  for env in envs: 
    for i in range(NUM_PASSES):
      cycle = env.get_cycles([i])
      cycles[i] += cycle
  print(cycles)
  return np.argsort(cycles)

# Take top N tested passes to run the insertion for the new pass 
def runInsertionN(envs, N, length = 10000, sort=False):
  #env = Env(pgm)  
  pool = ThreadPool(len(envs))
  sample_size = 0
  top_passes = [[] for idx in range(N)]
  top_timings = [10000000.0 for idx in range(N)]

  i = 0 # Walk through the pass pool 3 times 
  j = 0 # If the top performance is not changing, just iterate thru the pass pool 1 more time
  pass_len = 0
  if sort:
    ranked_pass_indices = get_lookup_rank(envs)
  else:
    ranked_pass_indices = np.arange(NUM_PASSES) 
  
  print(ranked_pass_indices) 
  # Foreach pass we insert it to existing passes 
  all_passes = getcycle.qw(getcycle.opt_passes_str)
  all_passes = all_passes[0:NUM_PASSES]
  total_passes = len(all_passes)
  print("total_passes_len: {}".format(total_passes))

  #while (i < total_passes * 3 and j < total_passes and pass_len < length): 
  begin = time.time()
  while (pass_len < length): 
    async_result = []
    pass_record = []
    time_record = []
    #pass_record.extend(top_passes)
    #time_record.extend(top_timings)

    #cur_pass = i % total_passes
    for k in range(N):
      for cur_pass in range(total_passes):
        (passes, wall_time, sample_count) = runInsertion(envs, top_passes[k], cur_pass, pool, ranked_pass_indices)
        pass_record.extend(passes)
        time_record.extend(wall_time)
        sample_size += sample_count 

# Multi processes implementation
# for j in range(N):
#     async_result.append(pool.apply_async(runInsertion, (pgm, top_passes[j], cur_pass)))
# for j in range(N):
#     (passes, wall_time) = async_result[j].get()
#     # Get current passes and the top passes from prev iter
#     pass_record.extend(passes)
#     time_record.extend(wall_time)

    print("pass_record: {}".format(pass_record))
    print("pass_timing: {}".format(time_record))
    print("sample_size: {}".format(sample_size))
    np_time_record = np.array(time_record)
    indices = np_time_record.argsort().flatten()[:N]

    print("indices: {}".format(indices))
    top_passes = [pass_record[idx] for idx in indices]
    new_top_timings = [time_record[idx] for idx in indices]

    pass_len = len(top_passes[0])

    end = time.time()
    print('Time: {}'.format(end-begin))
    print('Top passes: {}'.format(top_passes))
    print('Top passes timing: {}'.format(new_top_timings))
    improve = False 
    for idx in range(len(new_top_timings)):
      if (new_top_timings[idx] < top_timings[idx]):
        improve = True      

      top_timings[idx] = new_top_timings[idx]


    #if not improve:
    #  j += 1
    i+=1
    #if i % total_passes== 0:
    #  walk = i / total_passes
    #  print("Round %d start"%walk)
  return (top_passes, top_timings, sample_size)

def runInsertion(envs, cur_passes, new_pass, pool, indices):
  pass_record = []
  time_record = []
  sample_count = 0 
  for j in range(len(cur_passes)+ 1):
    test_passes = np.insert(cur_passes, j, new_pass).astype(int).tolist()
    test_passes = np.take(indices, test_passes)
    #print('Test passes: {}.'.format(test_passes))
    #_, reward = env.reset(init=test_passes)
    rews = pool.map(lambda env: env.get_cycles(test_passes), envs)
    sample_count += len(envs) 
    reward = -geo_mean(rews)

    wall_time = -reward
    pass_record.append(test_passes)
    time_record.append(wall_time)
  return (pass_record, time_record, sample_count)

def in_test_single_pgm(N=1):
  test_len = [12]
  fout = open("report_in_12.txt", "w")
  fout.write("Benchmark | Cycle Counts | Algorithm Runtime (s)| Sample Sizes | Passes \n")

  from gym_hls.envs.chstone_bm import get_all9
  bms = get_all9() 
  for bm in bms: 
    for length in test_len:
      envs = []
      i = 0
      pgm, path  = bm
      print("Program: {}".format(pgm))
      env_config = {
        'pgm':pgm,
        'pgm_dir':path,
        'run_dir':'run_'+pgm.replace(".c",""),
        'normalize':False,
        'orig_and_normalize':False,
        'log_obs_reward':False,
        'verbose':False,
        }
      envs.append(Env(env_config))
      i = i+1
      begin = time.time()
      (passes, timings, sample_size) = runInsertionN(envs, N, length=length, sort=True)
      end = time.time()
      print("Best individuals are: {}".format(passes[0]))
      print("Cycles: {}".format(timings[0]))
      compile_time =end - begin
      print("Compile Time: %d"%(int(compile_time)))
      fout.write("{}|{}|{}|{}|{}".format(pgm, timings[0], compile_time, sample_size, passes[0]))

def in_test_pgm_group(N=1):
  test_len = [12]
  for length in test_len:
    print("Program: {}".format(pgm))
    #env=Env(pgm, path)  
    from gym_hls.envs.chstone_bm import get_all9
    bms = get_all9() 

    envs = []
    i = 0
    for pgm, path in bms:
      env_config = {
        'pgm':pgm,
        'pgm_dir':path,
        'run_dir':'run_'+pgm.replace(".c",""),
        'normalize':False,
        'orig_and_normalize':False,
        'log_obs_reward':False,
        'verbose':False,
        }

      envs.append(Env(env_config))
      i = i+1

    begin = time.time()
    (passes, timings, sample_size) = runInsertionN(envs, N, length=length)
    end = time.time()
    #pickle.dump(record, lb_file)
    print("Best individuals are: {}".format(passes[0]))
    print("Cycles: {}".format(timings[0]))
    compile_time =end - begin
    print("Compile Time: %d"%(int(compile_time)))
    fout.write("{}|{}|{}|{}|{}\n".format("get_all9", timings[0], compile_time, sample_size, passes[0]))



if __name__== "__main__":
  in_test_single_pgm()

