import pickle

import numpy as np
def geomean(iterable):
    a = np.array(iterable).astype(float)
    prod = a.prod()
    prod = -prod if prod < 0 else prod
    return prod**(1.0/len(a))

def parse_rollout(baseline_fn="baseline.txt", rollout_fn="ppo_results_orig_norm_24pass_random_log.csv"): 
  results = {}

  total_count = 0

  total_o3_cycle = []
  with open(rollout_fn) as f:
    lines = f.readlines()
    for line in lines:
      data = line.split(',') 
      pgm = data[0] + '.c'
      cycle = int(data[1].replace('\n',''))
      if cycle < 10000000:
        cycles = [cycle]
        results[pgm] = cycles
        total_count += 1
        total_o3_cycle.append(cycle)
  

  better_count = 0
  equal_count = 0
  total_rl_cycle = []
  with open(baseline_fn) as f:
    lines = f.readlines()
    lines = lines[1:]
    for line in lines:
      data = line.split('|') 
      if data[0] in results:
        cycle = int(data[2])
        results[data[0]].append(cycle)
        total_rl_cycle.append(cycle)
        if cycle == 10000000:
          print(data[0])
          raise 
        if cycle < results[data[0]][0]: 
          better_count += 1
        if cycle == results[data[0]][0]:
          better_count += 1

  print(results)
  print("total_count: {}".format(total_count))
  print("better_count: {}".format(better_count))
  print("total_count: {}".format(equal_count))
  print("worse_count: {}".format(total_count - better_count - equal_count))
  avg_o3_cycle = np.average(total_o3_cycle)
  avg_rl_cycle = np.average(total_rl_cycle)
  geomean_o3_cycle = geomean(total_o3_cycle)
  geomean_rl_cycle = geomean(total_rl_cycle)
  print("average o3 cycles: {}".format(avg_o3_cycle))
  print("average rl cycles: {}".format(avg_rl_cycle))
  print("ratio: {}".format(avg_o3_cycle/avg_rl_cycle))
  print("geomean o3 cycles: {}".format(geomean_o3_cycle))
  print("geomean rl cycles: {}".format(geomean_rl_cycle))
  print("ratio: {}".format(geomean_o3_cycle/geomean_rl_cycle))
  
parse_rollout()
