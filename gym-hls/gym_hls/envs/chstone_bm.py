# Returns list of (pgm, pgm_dir) tuple
# Returns list of (pgm, pmg_files) tuple
from gym_hls.envs.utils import lsFiles
import os

def get_random(path="/scratch/qijing.huang/random_pgm/dataset", N=10):
  """
  Examples :
    >>> print(get_random())
    [file1.txt, file2.txt, file3.txt, file4.txt, file5.txt, file6.txt, file7.txt, file8.txt, file9.txt, file10.txt]
    >>> print(get_random(“path/to/directory”))
    [fl1.txt, fl2.txt, fl3.txt, fl4.txt, fl5.txt, fl6.txt, fl7.txt, fl8.txt, fl9.txt, fl10.txt]
    >>> print(get_random(“path/to/directory”, 4))
    [fl1.txt, fl2.txt, fl3.txt, fl4.txt]

  Args:
    path (str, optional): The path of the directory we are interested in. Defaults to "/scratch/qijing.huang/random_pgm/dataset".
    N (int, optional): N is the number of benchmarks to pick from the given path. Defaults to ten.

  Returns:
    Returns a list of N strings where each element is the path to a benchmark file.
  """

  pgms = lsFiles(path)
  random_list = []
  for i in range(N):
    random_list.append(pgms[i], path) 
  print(random_list)
  return random_list 
 
chstone_path = os.environ["LEGUP_PATH"]+"/examples/chstone/"  
def get_chstone(path= chstone_path, N=12, use_dir=True):
  """
  Examples :
    >>> print(get_chstone())
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"), ("dfadd.c","path/dfadd/"),("dfdiv.c","path/dfdiv/"), 
    ("dfmul.c","path/dfmul/"), ("dfsin.c","path/dfsin/"), ("gsm.c","path/gsm/"), ("main.c","path/jpeg/"), ("mips.c","path/mips/"), 
    ("mpeg2.c","path/motion/"), ("sha_driver.c","path/sha/")]   
    >>> print(get_random(“path/to/dir/”))
    [("adpcm.c","path/to/dir/adpcm/"), ("aes.c","path/to/dir/aes/"), ("bf.c","path/to/dir/blowfish/"), ("dfadd.c","path/to/dir/dfadd/"), 
    ("dfdiv.c","path/to/dir/dfdiv/"),  ("dfmul.c","path/to/dir dfmul/"), ("dfsin.c","path/to/dir/dfsin/"), ("gsm.c","path/to/dir/gsm/"), 
    ("main.c","path/to/dir/jpeg/"), ("mips.c","path/to/dir/mips/"), ("mpeg2.c","path/to/dir/motion/"), ("sha_driver.c","path/to/dir/sha/")]   

    >>> print(get_random(“path/to/directory”, 6))
    [("adpcm.c","path/to/dir/adpcm/"), ("aes.c","path/to/dir/aes/"), ("bf.c","path/to/dir/blowfish/"), ("dfadd.c","path/to/dir/dfadd/"), 
    ("dfdiv.c","path/to/dir/dfdiv/"),  ("dfmul.c","path/to/dir dfmul/")]

    >>> print(get_random(“path/to/directory”, 6, False))
    [("adpcm.c",["path/to/dir/adpcm/"]), ("aes.c",["path/to/dir/aes/"]), ("bf.c",["path/to/dir/blowfish/"]), ("dfadd.c",["path/to/dir/dfadd/"]), 
    ("dfdiv.c",["path/to/dir/dfdiv/"]),  ("dfmul.c",["path/to/dir dfmul/"])]

  Args:
    path (str, optional): Path to the chstone_path directory that contains chstone benchmarks. Defaults to chstone_path.
    N (int, optional): N is the number of benchmarks to select from the chstone list. Defaults to twelve.
    use_dir(bool, optional): use_dir should be set to True if you want the tuple path (given path + “benchmark_name”) in the returned list (chstone_list) 
      to as a string, or use_dir should be set to False if you want the tuple path (given path + “benchmark_name”) in the returned list (chstone_list) to 
      be as a string in a list. Defaults to True.

  Returns:
      Returns a list of tuples where each tuple(“string”, [“string”] ) contains as the first element a file written in the C programming language(chstone_benchmark_name.c from the chstone list) and as the second element a list that contains the same chstone benchmark name (case when use_dir is False). However, for the case when use_dir is True, this function returns a list of tuples where each tuple(“string”, “string”) contains as the first element a file written in the C programming language (chstone_benchmark_name.c from the chstone list)  and as the second element the same chstone benchmark name concatenated with the given path.
  """

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

def get_others(path= os.environ["LEGUP_PATH"] + "/examples/", use_dir=True):
  """
  Examples :
    >>> print(get_others())
    [("fir.c","path/fir/"), ("dhry.c","path/dhrystone/"), ("qsort.c","path/qsort/"), ("matrixmultiply.c","path/matrixmultiply/”)]   
          
    >>> print(get_others(“path/to/dir/”))
    [("fir.c","path/to/dir/fir/"), ("dhry.c","path/to/dir/dhrystone/"), ("qsort.c","path/to/dir/qsort/"), ("matrixmultiply.c","path/to/dir/matrixmultiply/”)]   

    >>> print(get_others(“path/to/directory”, False))
    [("fir.c",["path/to/dir/fir/"]), ("dhry.c",["path/to/dir/dhrystone/"]), ("qsort.c",["path/to/dir/qsort/"]), ("matrixmultiply.c",["path/to/dir/matrixmultiply/”])]   

  Args:
      path (str, optional): Path to a directory that contains other benchmarks (besides the ones available at chstone). 
        Defaults to  os.environ["LEGUP_PATH"] + "/examples/".
      Use_dir (bool, optional): use_dir should be set to True if you want the tuple path (given path + “other_benchmark_name”) in the returned list (others_list) 
        to as a string, or use_dir should be set to False if you want the tuple path (given path + “other_benchmark_name”) in the returned list (others_list) to be 
        as a string in a list. Defaults to True.

  Returns:
    Returns a list of tuples where each tuple(“string”, [“string”] ) contains as the first element a file written in the C programming language(other_benchmark_name.c 
    from the others list) and as the second element a list that contains the same other benchmark name (case when use_dir is False). However, for the case when use_dir 
    is True, this function returns a list of tuples where each tuple(“string”, “string”) contains as the first element a file written in the C programming language 
    (other_benchmark_name.c from the others list)  and as the second element the same other_benchmark_name concatenated with the given path.

  """

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
  """
  Examples :
    >>> print(get_gsm())
    [("gsm.c","path/gsm/")]

  Returns:
    Returns a list that contains a tuple of two strings where the first element of the tuple represents a file written in C programming language 
    (the chstone benchmark gsm.c) and the second element represents the path to the gsm file.
  """

  bm = list (get_chstone()[i] for i in [7])
  print(bm)
  return bm

def get_orig4():
  """
  Examples :
    >>> print(get_orig4())
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"),("gsm.c","path/gsm/")]

  Returns:
    Returns a list that contains 4 tuples where the first element of the each tuple(“string”, “string”) represents a file written in C programming language 
    (the chstone_benchmark_name.c) and the second element represents the path to the same chstone benchmark file. The 4 chstone benchmarks that this list contain are: 
    adpcm, aes, blowfish, and gsm.
  """

  bm = list (get_chstone()[i] for i in [0, 1, 2, 7])
  print(bm)
  return bm

def get_orig6():
  """
  Examples :
    >>> print(get_orig6())
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"), ("gsm.c","path/gsm/"), ("mpeg2.c","path/motion/"), ("sha_driver.c","path/sha/")]   

  Returns:
    Returns a list that contains 6 tuples where the first element of the each tuple(“string”, “string”) represents a file written in C programming language 
    (the chstone_benchmark_name.c) and the second element represents the path to the same chstone benchmark file. The 6 chstone benchmarks that this list contain are: 
    adpcm, aes, blowfish, gsm, motion and sha.
  """

  bm = list (get_chstone()[i] for i in [0, 1, 2, 7, 10, 11])
  print(bm)
  return bm

def get_ot6():
  """
  Examples :
    >>> print(get_ot6())
    [("dfadd.c","path/dfadd/"), ("main.c","path/jpeg/"), ("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"),("dfadd.c","path/dfadd/")] 

  Returns:
    Returns a list that contains 6 tuples where the first element of the each tuple(“string”, “string”) represents a file written in C programming language 
    (the chstone_benchmark_name.c) and the second element represents the path to the same chstone benchmark file. The 5 chstone benchmarks that this list contain are: 
    dfadd, jpeg, adpcm, aes, and blowfish.
  """

  bm = list (get_chstone()[i] for i in [3, 8])
  ot = list (get_others()[i] for i in [0, 1, 2, 3])
  bm.extend(ot)
  print(bm)
  return bm

def get_all9():
  """
  Examples :
    >>> print(get_all9())
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"), ("gsm.c","path/gsm/") , ("mpeg2.c","path/motion/"), ("sha_driver.c","path/sha/"), 
    ("aes.c","path/aes/"), ("bf.c","path/blowfish/"),("dfadd.c","path/dfadd/")] 

  Returns:
    Returns a list that contains 9 tuples where the first element of the each tuple(“string”, “string”) represents a file written in C programming language 
    (the chstone_benchmark_name.c) and the second element represents the path to the same chstone benchmark file. The 6 chstone benchmarks that this list contain are: 
    adpcm, aes, blowfish, gsm motion, and dfadd
  """

  bm = list (get_chstone()[i] for i in [0, 1, 2, 7, 10, 11])
  ot = list (get_others()[i] for i in [1, 2, 3])
  bm.extend(ot)
  bm.sort(key=lambda x: x[0])
  print("get_all9 with %d programs"%len(bm))
  print(bm)
  return bm

def get_all12():
  """
  Examples :
    >>> print(get_all12())
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"), ("dfadd.c","path/dfadd/"), ("gsm.c","path/gsm/"),("main.c","path/jpeg/"),
    ("mpeg2.c","path/motion/"), ("sha_driver.c","path/sha/"), ("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"),("dfadd.c","path/dfadd/")]

  Returns:
    Returns a list that contains 12 tuples where the first element of the each tuple(“string”, “string”) represents a file written in C programming language 
    (the chstone_benchmark_name.c) and the second element represents the path to the same chstone benchmark file. The 8 chstone benchmarks that this list contain are: 
    adpcm, aes, blowfish, dfadd, gsm, jpeg, motion, and sha. The elements of the list are sorted in alphabetical order.
  """

  bm = list (get_chstone()[i] for i in [0, 1, 2, 3, 7, 8, 10, 11])
  ot = list (get_others()[i] for i in [0, 1, 2, 3])
  bm.extend(ot)
  print(bm)
  return bm

def get_bms(test_name):
  """
  Examples :
    >>> print(get_bms(“orig4”))
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"),("gsm.c","path/gsm/")]

    >>> print(get_bms(“all9”))
    [("adpcm.c","path/adpcm/"), ("aes.c","path/aes/"), ("bf.c","path/blowfish/"), ("gsm.c","path/gsm/") , ("mpeg2.c","path/motion/"), ("sha_driver.c","path/sha/"), 
    ("aes.c","path/aes/"), ("bf.c","path/blowfish/"),("dfadd.c","path/dfadd/")]

  Args:
      test_name(str): test_name can be any of the following names that indicates to one of the functions defined (above) in this class: orig4, orig6, ot6, all12, and gsm
  
  Raises:
      Value Error:  if test_name is not equal to any of the different names mentioned above.     

  Returns:
    Returns a list of tuples where the first element of the each tuple(“string”, “string”) represents a file written in C programming language 
    (the chstone_benchmark_name.c) and the second element represents the path to the same chstone benchmark file. The list can contain different number of 
    benchmarks depending on the function called. 
  """

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
