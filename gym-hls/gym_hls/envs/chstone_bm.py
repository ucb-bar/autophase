# Returns list of (pgm, pgm_dir) tuple
def get_chstone(path= "/scratch/qijing.huang/LegUp/legup-4.0/examples/chstone/", N=12):
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
  #for key, value in chstone.items():
  for key, value in chstone:
    chstone_list.append((value+".c", path+key+"/"))

  return chstone_list[:N]

def get_others(path="/scratch/qijing.huang/LegUp/legup-4.0/examples/"):
  others = [
  ("fir" ,"fir"),
  ("dhrystone" ,"dhry"),
  ("qsort" ,"qsort"),
  ("matrixmultiply" ,"matrixmultiply")]

  #"fft" :"fft",
  #"shift" :"shift",
  #"llist" :"llist"

  others_list = []
  #for key, value in others.items():
  for key, value in others:
    others_list.append((value+".c", path+key+"/"))

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

