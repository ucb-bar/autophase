import pickle
import sys

import numpy as np
def geomean(iterable):
    a = np.array(iterable).astype(float)
    prod = a.prod()
    prod = -prod if prod < 0 else prod
    return prod**(1.0/len(a))

# Define the valid programs here
def is_valid_pgm(pgm):
  pgms = ['471', '4926', '12092', '3449', '4567', '16510', '6118', '15427', '112', '15801', '3229', '12471', '3271', '16599', '11090', '16470', '10308', '9724', '8971', '15292', '15117', '6827', '9381', '18028', '4278', '16971', '1985', '12721', '16698', '7246', '1335', '7923', '13570', '11580', '16010', '10492', '10396', '13085', '17532', '14602', '16879', '8518', '1546', '12204', '15008', '5381']
  for ref_pgm in pgms: 
      if pgm == ref_pgm:
          return True 
  return False

def parse_rollout(baseline_fn="baseline.txt", rollout_fn="ppo_results_orig_norm_24pass_random_log.csv"): 
  pgms = []
  results = {}

  total_count = 0

  total_rl_cycle = []
  with open(rollout_fn) as f:
    lines = f.readlines()
    for line in lines:
      data = line.split(',') 
      pgm = data[0] + '.c'
      cycle = int(data[1].replace('\n',''))
      #if cycle < 20000 and cycle > 1000:
      #if cycle < 10000000 and is_valid_pgm(data[0]):
      if cycle < 10000000:
        cycles = [cycle]
        results[pgm] = cycles
        total_count += 1
        total_rl_cycle.append(cycle)
        pgms.append(data[0])

  better_count = 0
  equal_count = 0
  total_o3_cycle = []
  with open(baseline_fn) as f:
    lines = f.readlines()
    lines = lines[1:]
    for line in lines:
      data = line.split('|') 
      if data[0] in results.keys():
        cycle = int(data[2])
        results[data[0]].append(cycle)
        total_o3_cycle.append(cycle)
        #if cycle == 10000000:
        #  print(data[0])
        #  raise 
        if cycle > results[data[0]][0]: 
          better_count += 1
        if cycle == results[data[0]][0]:
          equal_count += 1

  print(results)
  print("total_count: {}".format(total_count))
  print("better_count: {}".format(better_count))
  print("equal_count: {}".format(equal_count))
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
  
  #print(pgms)
if __name__ == '__main__':
    rollout_fn = sys.argv[1]
    parse_rollout(rollout_fn=rollout_fn)
