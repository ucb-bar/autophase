import gym_hls.envs.getox
import gym_hls.envs.getpgm
import subprocess
from  subprocess import call
import os
import re 

features = ["# of BB where total args for phi nodes > 5", "# of BB where total args for phi nodes is [1, 5]", "# of BB's with 1 predecessor", "# of BB's with 1 predecessor and 1 successor", "# of BB's with 1 predecessor and 2 successors", "# of BB's with 1 successor", "# of BB's with 2 predecessors", "# of BB's with 2 predecessors and 1 successor", "# of BB's with 2 predecessors and successors", "# of BB's with 2 successors", "# of BB's with >2 predecessors", "# of BB's with Phi node # in range (0, 3]", "# of BB's with more than 3 Phi nodes", "# of BB's with no Phi nodes", "# of Phi-nodes at beginning of BB", "# of branches", "# of calls that return an int", "# of critical edges", "# of edges", "# of occurrences of 32-bit integer constants", "# of occurrences of 64-bit integer constants", "# of occurrences of constant 0", "# of occurrences of constant 1", "# of unconditional branches", "Binary operations with a constant operand", "Number of AShr insts", "Number of Add insts", "Number of Alloca insts", "Number of And insts", "Number of BB's with instructions between [15, 500]", "Number of BB's with less than 15 instructions", "Number of BitCast insts", "Number of Br insts", "Number of Call insts", "Number of GetElementPtr insts", "Number of ICmp insts", "Number of LShr insts", "Number of Load insts", "Number of Mul insts", "Number of Or insts", "Number of PHI insts", "Number of Ret insts", "Number of SExt insts", "Number of Select insts", "Number of Shl insts", "Number of Store insts", "Number of Sub insts", "Number of Trunc insts", "Number of Xor insts", "Number of ZExt insts", "Number of basic blocks", "Number of instructions (of all types)", "Number of memory instructions", "Number of non-external functions", "Total arguments to Phi nodes", "Unary"] 

def run_stats(bc_code, path="."):
    """
    Examples :
        >>> print(run_stats(bc_code, path="."))
        [path/file1.txt, path/file2.txt, path/file3.txt]

    Args:
        bc_code (str): bc_code is the name of a bitcode file.
        path (str): This is the path to the directory we are interested in. Defaults to the directory.
    Returns:
        Returns a list of all the features extracted from the bitcode file, which is the compiled version of program we need to run, that exist in our features 
        list (that we assigned in this class). 
    """

    opt_path = os.environ["LEGUP_PATH"] + "/llvm/Release+Asserts/bin/"
    cmd = opt_path + "opt -stats -instcount " + bc_code + " "
    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=path)
    (out, err) = proc.communicate()
    #print (err)
    m = parse_static_features_str(err)
    return m
    

def parse_static_features_str(out):
    """ 
    Args:
        out (str): is an error message from Popen.communicate()

    Returns:
        Returns a list of all the features extracted from the bitcode file, which is the compiled version of program we need to run, that exist in our features 
        list(This is a helper function for run_stats and both functions return the same list).

    """

    feat_ids = []
    for feature in features: 
      my_regex = r"\s*(\d+) instcount - " + re.escape(feature)
      p = re.compile(my_regex) 
      m = p.findall(out.decode("utf-8"))
      if len(m):
        feat_ids.append(int(m[0]))
      else:
        feat_ids.append(0) 
      
    return feat_ids


#parse_static_features("static_features.txt")

def get_pgm_list():
    """
    Examples :
        >>> print(get_pgm_list())
        [file1.txt]

    Returns:
        returns one program name from the dataset directory.
    """

    pgms = getpgm.lsFiles()
    pgms = pgms[0:1]
    return pgms 

def main ():
    """
    The main function select a program, compiles it with -O3 and without any flags, then write the features of both compiled files in features_o3.txt and features.txt. 

    """

    fout = open("features.txt", "w")
    fout_o3 = open("features_o3.txt", "w")

    # Change to skeleton foler 
    os.chdir("./examples")
    pgms = ['gsm.c']

    #print(len(features))
    fout.write(str(features) + "\n")
    fout_o3.write(str(features) + "\n")
    for pgm in pgms: 
        # Copy to skeleton folder 
        # getpgm.copyFile(pgm, "../dataset", "../skeleton_o3")
        print("Program: %s" % pgm)

        c_code = pgm.replace('.c', '') 
        fout.write(c_code + "\t")
        fout_o3.write(c_code + "\t")
        # Compile the program with -O3
        geto3.getOxCycles(c_code) 

        bc_code = c_code + ".prelto.bc" 
        bc_code_o3 = c_code + ".bc"

        feat = run_stats(bc_code)
        feat_o3 = run_stats(bc_code_o3) 

        fout.write(str(feat) + "\n")
        fout_o3.write(str(feat_o3) + "\n")
                
        #getpgm.rmFile(pgm, "../skeleton_o3")
        #call(["make", "clean"])
    fout.close()
    fout_o3.close()

if __name__ == "__main__":
    main()

