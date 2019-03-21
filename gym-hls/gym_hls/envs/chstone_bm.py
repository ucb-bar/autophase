# Returns list of (pgm, pgm_dir) tuple
# Returns list of (pgm, pmg_files) tuple
from gym_hls.envs.utils import lsFiles

def get_random(path="/scratch/qijing.huang/random_pgm/dataset", N=10):
    pgms = lsFiles(path)
    random_list = []
    for i in range(N):
      random_list.append(pgms[i], path) 
    print(random_list)
    return random_list 
   
def get_chstone(path= "/scratch/qijing.huang/LegUp/legup-4.0/examples/chstone/", N=12, use_dir=True):
  chstone = [
 ( "adpcm","adpcm"),
 ( "aes","aes"),
 ( "blowfish","bf"),
 ( "dfadd","dfadd"),
 ( "dfdiv","dfdiv"),
 ( "dfmul","dfmul"),
 ( "dfsin","dfsin"),
 ( "gsm","gsm"),
 ( "jpeg","main"),
 ( "mips","mips"),
 ( "motion","mpeg2"),
 ( "sha","sha_driver")
  ]

  chstone_list = []
  for key, value in chstone:
    if use_dir: 
      chstone_list.append((value+".c", path+key+"/"))
    else: 
      files = lsFiles(path+key) 
      chstone_list.append((value+".c", files))
  return chstone_list[:N]

def get_others(path="/scratch/qijing.huang/LegUp/legup-4.0/examples/", use_dir=True):
  others = [
  ("fir" ,"fir"),
  ("dhrystone" ,"dhry"),
  ("qsort" ,"qsort"),
  ("matrixmultiply" ,"matrixmultiply")]

  others_list = []
  for key, value in others:
    if use_dir: 
      others_list.append((value+".c", path+key+"/"))
    else:
      files = lsFiles(path+key) 
      others_list.append((value+".c", files))
  return others_list


def get_gsm():
  bm = list (get_chstone()[i] for i in [7])
  print(bm)
  return bm

def get_orig4():
  bm = list (get_chstone()[i] for i in [0, 1, 2, 7])
  print(bm)
  return bm

def get_orig6():
  bm = list (get_chstone()[i] for i in [0, 1, 2, 7, 10, 11])
  print(bm)
  return bm

def get_ot6():
  bm = list (get_chstone()[i] for i in [3, 8])
  ot = list (get_others()[i] for i in [0, 1, 2, 3])
  bm.extend(ot)
  print(bm)
  return bm

def get_all9():
  bm = list (get_chstone()[i] for i in [0, 1, 2, 7, 10, 11])
  ot = list (get_others()[i] for i in [1, 2, 3])
  bm.extend(ot)
  bm.sort(key=lambda x: x[0])
  print("get_all9 with %d programs"%len(bm))
  print(bm)
  return bm

def get_all12():
  bm = list (get_chstone()[i] for i in [0, 1, 2, 3, 7, 8, 10, 11])
  ot = list (get_others()[i] for i in [0, 1, 2, 3])
  bm.extend(ot)
  print(bm)
  return bm

def get_bms(test_name):
  if test_name == "orig4":
    return get_orig4()
  elif test_name == "orig6":
    return get_orig6()
  elif test_name == "ot6":
    return get_ot6()
  elif test_name == "all12":
    return get_all12()
  elif test_name == "gsm":
    return get_gsm()
  else:
    raise Exception("Please specify a benchmark!")

if __name__ == "__main__":
  get_all9()
