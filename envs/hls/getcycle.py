import re
import subprocess

# Available LLVM optimizatons
# tailduplicate, simplify-libcalls, -block-placement  

#opt_passes_str="-inline -jump-threading -simplifycfg -gvn -loop-rotate -codegenprepare"
opt_passes_str = "-correlated-propagation -scalarrepl -lowerinvoke -strip -strip-nondebug -sccp -globalopt -gvn -jump-threading -globaldce -loop-unswitch -scalarrepl-ssa -loop-reduce -break-crit-edges -loop-deletion -reassociate -lcssa -codegenprepare -memcpyopt -functionattrs -loop-idiom -lowerswitch -constmerge -loop-rotate -partial-inliner -inline -early-cse -indvars -adce -loop-simplify -instcombine -simplifycfg -dse -loop-unroll -lower-expect -tailcallelim -licm -sink -mem2reg -prune-eh -functionattrs -ipsccp -deadargelim -sroa -loweratomic -terminate"

    

# This is not used before extra dependency in Makefile.config need to be set 
compile_str = """
../../mark_labels.pl test_c_code.c > test_c_code_labeled.c
clang-3.5 test_c_code_labeled.c -emit-llvm -c -fno-builtin -I ../../lib/include/ -m32 -I /usr/include/i386-linux-gnu -O0 -fno-vectorize -fno-slp-vectorize -o test_c_code.prelto.1.bc
cp -f test_c_code.prelto.1.bc test_c_code.prelto.cv.bc
../../../llvm/Release+Asserts/bin/opt opt_passes < test_c_code.prelto.cv.bc > test_c_code.prelto.2.bc
cp test_c_code.prelto.2.bc test_c_code.prelto.linked.bc
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -std-link-opts < test_c_code.prelto.linked.bc -o test_c_code.prelto.linked.1.bc
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -legup-prelto < test_c_code.prelto.linked.1.bc > test_c_code.prelto.6.bc
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -std-link-opts < test_c_code.prelto.6.bc -o test_c_code.prelto.bc
../../../llvm/Release+Asserts/bin/llvm-link  test_c_code.prelto.bc ../../lib/llvm/liblegup.bc ../../lib/llvm/libm.bc -o test_c_code.postlto.6.bc
../../../llvm/Release+Asserts/bin/opt -internalize-public-api-list=main -internalize -globaldce test_c_code.postlto.6.bc -o test_c_code.postlto.8.bc
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -instcombine -std-link-opts < test_c_code.postlto.8.bc -o test_c_code.postlto.bc
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -basicaa -loop-simplify -indvars2  -loop-pipeline test_c_code.postlto.bc -o test_c_code.1.bc
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -instcombine test_c_code.1.bc -o test_c_code.bc 
../../../llvm/Release+Asserts/bin/llc -legup-config=../../legup.tcl  -march=v test_c_code.bc -o test_c_code.v
../../../llvm/Release+Asserts/bin/opt -load=../../../llvm/Release+Asserts/lib/LLVMLegUp.so -legup-config=../../legup.tcl  -legup-track-bb < test_c_code.bc > test_c_code.track_bb.bc
../../../llvm/Release+Asserts/bin/llvm-dis test_c_code.track_bb.bc
../../../llvm/Release+Asserts/bin/llc -march=x86-64 test_c_code.track_bb.bc
gcc test_c_code.track_bb.s -o test_c_code.track_bb
./test_c_code.track_bb | grep 'Track@' | sed 's/Track@//' > test_c_code.lli_bb.trace
rm test_c_code.track_bb
perl ../../../tiger/tool_source/profiling_tools/../partition_analysis/get_hw_cycle.pl test_c_code.lli_bb.trace test_c_code.acel_cycle.rpt
"""

# Generate makefile instead, need to modify Makefile.common 119 
##  ifdef CUSTOM_OPT
##  	$(LLVM_HOME)opt $(EXTRA_OPT_FLAGS) < $(NAME).prelto.cv.bc > $(NAME).prelto.2.bc
##  else # CUSTOM_OPT
##  ifdef UNROLL
##  	$(LLVM_HOME)opt -mem2reg -loops -loop-simplify -loop-unroll $(UNROLL) < $(NAME).prelto.cv.bc > $(NAME).prelto.2.bc
##  else # UNROLL
##  ifeq ($(DEBUG_KEEP_VARS_IN_MEM),1)
##  	$(LLVM_HOME)opt -loops -loop-simplify < $(NAME).prelto.cv.bc > $(NAME).prelto.2.bc
##  else  # DEBUG_KEEP_VARS_IN_MEM
##  	$(LLVM_HOME)opt -mem2reg -loops -loop-simplify < $(NAME).prelto.cv.bc > $(NAME).prelto.2.bc
##  endif # DEBUG_KEEP_VARS_IN_MEM
##  endif # UNROLL
##  endif # CUSTOM_OPT
makefile_str= """
NAME= test_c_code
NO_OPT=1
CUSTOM_OPT=1
EXTRA_OPT_FLAGS = opt_passes
LEVEL = /scratch/qijing.huang/LegUp/legup-4.0/examples
include $(LEVEL)/Makefile.common
"""
def qw(s):
        return tuple(s.split())

def countPasses():
    count=len(qw(opt_passes_str))
    return count
    
# Get a tuple of optimizations
def getPasses(opt_indice):
    return map((lambda x: opt_passes[x]), opt_indice)

opt_passes = qw(opt_passes_str)

def passes2indice(passes):
    indices = []
    passes = qw(passes)
    for passs in passes:
      for i in range(len(opt_passes)):
        if passs == opt_passes[i]:
          indices.append(i)
          break
    #print(indices)
    return indices

def getHWCycles(c_code, opt_indice, path=".", sim=False):
    #print(len(opt_passes))
    ga_seq = getPasses(opt_indice)
    ga_seq_str = " ".join(ga_seq)
    #print("Passes: %s"%ga_seq_str)

    makefile_new = makefile_str.replace("test_c_code", c_code) 
    makefile_new = makefile_new.replace("opt_passes", ga_seq_str) 
    #print(makefile_new)
    # Update the Makefile 
    f = open( path+"/Makefile","w")
    f.write(makefile_new)
    f.close()    
    done = False
    # Run HLS
    #proc = subprocess.Popen(["make accelerationCycle -s"], stdout=subprocess.PIPE, shell=True)
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
      proc = subprocess.Popen(["make clean accelerationCycle -s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=path)
      (out, err) = proc.communicate()
      #print(err)
      #if (err):
      #    print (err)
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
          if int(hw_cycle) == 0:
            hw_cycle = 10000000
            done = True
      else:
          hw_cycle = 10000000 # problematic 
          done = True
      #print("Cycles: %s"%hw_cycle)
    return int(hw_cycle), done

def main():
  #indices=[16, 25, 6, 10, 23, 22, 18, 11, 31, 34, 27, 33]
  #indices=[14, 22, 23]
  indices=[23, 9, 31, 0, 25, 30]
  passes=getPasses(indices)
  passes_str =" ".join(str(x) for x in passes)
  print(passes_str)

if __name__ == "__main__":
  main()

