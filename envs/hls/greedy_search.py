import time
import getcycle
from env import Env
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

# Take top N tested passes to run the insertion for the new pass 
def runInsertionN(envs, N, length = 10000):
  #env = Env(pgm)  
  pool = ThreadPool(len(envs))

  top_passes = [[] for idx in range(N)]
  top_timings = [10000000.0 for idx in range(N)]

  i = 0 # Walk through the pass pool 3 times 
  j = 0 # If the top performance is not changing, just iterate thru the pass pool 1 more time
  pass_len = 0
# Foreach pass we insert it to existing passes 
  all_passes = getcycle.qw(getcycle.opt_passes_str)
  all_passes = all_passes[0:NUM_PASSES]
  total_passes = len(all_passes)
  print("total_passes_len: {}".format(total_passes))

  #while (i < total_passes * 3 and j < total_passes and pass_len < length): 

  begin = time.time()
  while ( pass_len < length): 
    async_result = []
    pass_record = []
    time_record = []
    #pass_record.extend(top_passes)
    #time_record.extend(top_timings)

    #cur_pass = i % total_passes
    for k in range(N):
      for cur_pass in range(total_passes):
        (passes, wall_time) = runInsertion(envs, top_passes[k], cur_pass, pool)
        pass_record.extend(passes)
        time_record.extend(wall_time)

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
  return (top_passes, top_timings)

def runInsertion(envs, cur_passes, new_pass, pool):
  pass_record = []
  time_record = []
  for j in range(len(cur_passes)+ 1):
    test_passes = np.insert(cur_passes, j, new_pass).astype(int).tolist()
    #print('Test passes: {}.'.format(test_passes))
    #_, reward = env.reset(init=test_passes)
    rews = pool.map(lambda env: env.reset(init=test_passes, get_obs=False)[1], envs)
    #print(rews)
    reward = -geo_mean(rews)

    wall_time = -reward
    pass_record.append(test_passes)
    time_record.append(wall_time)
  return (pass_record, time_record)

def in1_test():

  from chstone_bm import get_chstone, get_others
  bm = get_chstone()
  bm.extend(get_others())
  test_len = [6, 12, 24, 48, 96]

  fout = open("report_in1.txt", "w")
  fout.write("Benchmark |Cycle Counts | Algorithm Runtime (s)|Passes \n")

  for pgm, path in bm:
    for length in test_len:

      begin = time.time()
      print("Program: {}".format(pgm))
      env=Env(pgm, path)  

      (passes, timings) = runInsertionN(env, 1, length=length)
      #pickle.dump(record, lb_file)
      print("Best individuals are: {}".format(passes[0]))
      print(timings)
      print("Cycles: {}".format(timings[0]))
      end = time.time()
      compile_time =end - begin
      print("Compile Time: %d"%(int(compile_time)))
    fout.write("{}|{}|{}|{}".format(pgm, timings[0], compile_time, passes[0]))

def in_test(N=1):

  #from chstone_bm import get_chstone, get_others
  #bm = get_chstone()
  #bm.extend(get_others())
  #test_len = [6, 12, 24, 48, 96]
  test_len = [12]

  fout = open("report_in_12.txt", "w")
  fout.write("Benchmark |Cycle Counts | Algorithm Runtime (s)|Passes \n")

  #for pgm, path in bm:
  for pgm in ["in"+str(N)+"_6progs"]:
    for length in test_len:

      begin = time.time()
      print("Program: {}".format(pgm))
      #env=Env(pgm, path)  
      from chstone_bm import get_bms
      bm = get_bms("orig6")

      envs = []
      i = 0
      for pg, path in bm:
          envs.append(Env(pg, path, "run_in_"+str(i)))
          i = i+1

      (passes, timings) = runInsertionN(envs, 1, length=length)
      #pickle.dump(record, lb_file)
      print("Best individuals are: {}".format(passes[0]))
      print(timings)
      print("Cycles: {}".format(timings[0]))
      end = time.time()
      compile_time =end - begin
      print("Compile Time: %d"%(int(compile_time)))
    fout.write("{}|{}|{}|{}".format(pgm, timings[0], compile_time, passes[0]))


if __name__== "__main__":
  in_test()

