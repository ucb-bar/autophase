import os 
from os.path import isfile, join
from shutil import copyfile

def lsFiles(path="../dataset"):
    path = os.path.abspath(path)
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    return files
    #print files

def copyFile(fn, src_path, dst_path):
    copyfile(join(src_path, fn), join(dst_path, fn))

    
def rmFile(fn, path):
    os.remove(join(path, fn))
