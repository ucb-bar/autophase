import glob 

def get_min(fn_prefix):
  files=glob.glob(fn_prefix+"*.log")

  sample_size = 0
  min_cycle = 100000000
  threads = len(files)
  actual_sample_size = 0

  for fn in files: 
    local_sample_size = 0
    with open(fn) as f:
      lines = f.readlines()
      for line in lines:
        data = line.split('|')
        if len(data) > 2:
          cycle = int(data[4])
          sample_size += 1
          local_sample_size += 1
          if cycle < min_cycle:
            actual_sample_size = local_sample_size
            min_cycle = cycle

  return (min_cycle, sample_size, actual_sample_size, threads)


def parse_log(algo="a3c"):
  from gym_hls.envs.chstone_bm import get_chstone, get_others, get_all9
  bms = get_all9()
  fout = open("report_"+algo+".txt", "w")
  for i, bm in enumerate(bms):
    pgm, _ = bm
    pgm = pgm.replace(".c", "")
    #pgm = str(i)
    min_cycle, sample_size, actual_sample_size, threads = get_min("run_{}_{}_".format(algo, pgm))
    fout.write("{}|{}|{}|{}|{}\n".format(pgm, min_cycle, sample_size, actual_sample_size, threads))

  fout.close()
  
parse_log("ppo")
