import pickle

def get_random_pgm_groups(): 
  interval = [0, 1000, 5000, 10000, 50000, 100000]
  buckets = {}
  for i in range(len(interval)-1):
    buckets[i] = []

  valid_count = 0
  with open('baseline.txt') as f:
    lines = f.readlines()
    print("Ignore Header Line: {}".format(lines[0]))
    lines = lines[1:]

    for line in lines:
      data = line.split('|') 
      bm  = data[0]
      o0_cycle = int(data[1])
      o3_cycle = int(data[2])

      if o0_cycle == 10000000:
        #print("{}".format(bm))
        continue 
   
      valid_count += 1
      for i in range(len(interval)-1):
        if o0_cycle >= interval[i] and o0_cycle < interval[i+1]:
          buckets[i].append(data)          
          break 

  print("Total {} valid programs".format(valid_count))
  for i in range(len(interval)-1):
    print("Interval {} ~ {}: {}".format(interval[i], interval[i+1],len(buckets[i])))

  output = open('random_pgms.pkl', 'wb')
  pickle.dump(buckets, output)
  output.close()

get_random_pgm_groups()
