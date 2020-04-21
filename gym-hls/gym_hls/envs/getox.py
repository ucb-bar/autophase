import re
import subprocess
import os

# Gather -O3 generated results
#makefile_str= """
#NAME= test_c_code
#LEVEL = /home/legup/legup-4.0/examples
#include $(LEVEL)/Makefile.common
#"""
makefile_str= """
NAME= test_c_code
NO_OPT=clang_opt
CUSTOM_OPT=1
EXTRA_OPT_FLAGS = test_flag\n"""+"LEVEL = " + os.environ["LEGUP_PATH"] + "/examples"+"""
include $(LEVEL)/Makefile.common
"""

def getOxCycles(c_code, path, level=3, clang_opt=False, sim=False):
  """
  Examples :
    >>> print(getOxCycles(c_code, path, level=3, clang_opt=False, sim=False)
    45

  Args:
    c_code (str): c_code is the name of the c code program we are optimizing.
    path (str): This is the path to the directory where the c_code file exists.
    level (int): This is an integer that represents different groups of optimizations implemented in the compiler. 
        Each optimization level is hand-picked by the compiler-designer to benefit specific benchmarks. Defaults to 3.
    sim (bool): sim is a Boolean that should be set to True if we want the subprocessor to run the “make clean p v -s” command, 
        and we should set it to False if we want the subprocessor to run the “make clean accelerationCycle -s” command instead. Defaults to False.
    clang_opt (bool): clang_opt is a Boolean that should be set to True if we want to use the clang option when running the HLS, and should be set to False otherwise.

  Returns:
    Returns the number of cycle counts it took to run the synthesized circuit made by using the passes set in the 0x optimization.

  """

  #print len(opt_passes)

  makefile_new = makefile_str.replace("test_c_code", c_code)
  if level > 0:
    makefile_new = makefile_new.replace("test_flag", "-O"+str(level))
  else:
    makefile_new = makefile_new.replace("test_flag", "")

  if clang_opt:
    makefile_new = makefile_new.replace("clang_opt", "0")
  else:
    makefile_new = makefile_new.replace("clang_opt", "1")

  # Update the Makefile
  f = open(path + "/Makefile","w")
  f.write(makefile_new)
  f.close()
  # Run HLS

  if sim:
    proc = subprocess.Popen(["make clean p v -s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=path)
    (out, err) = proc.communicate()
    print(out)
    #print(err)
    p = re.compile(r".*Cycles:\s+(\d+)", re.DOTALL)
    m = re.match(p, out.decode("utf-8") )
    if m:
        hw_cycle = m.group(1)
        if int(hw_cycle) == 0:
          hw_cycle = 10000000
    else:
        #print ("NM")
        hw_cycle = 10000000 # problematic 

  else:
    proc = subprocess.Popen(["make clean accelerationCycle -s"], stdout=subprocess.PIPE, shell=True, cwd=path)
    #proc = subprocess.Popen(["make accelerationCycle -s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    #if err:
    #    f = open(c_code+"err.trace", "w")
    #    f.write(err)
    #    f.close()

    #print "program output:", out
    #print "program error:", err

    p = re.compile(r"^.*main \|\s+(\d+).*", re.DOTALL)
    #p = re.compile(r'main')
    m = re.match(p, out.decode("utf-8") )
    # Parse Results
    if m:
        hw_cycle = m.group(1)
    else:
        print ("NM")
        hw_cycle = 10000000 # problematic
  print("Cycles: %s"%hw_cycle)
  return int(hw_cycle)



