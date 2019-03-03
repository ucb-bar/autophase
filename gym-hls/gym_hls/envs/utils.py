import os  
from os.path import isfile, join
from shutil import copyfile

def lsFiles(path, with_dir=True):
    path = os.path.abspath(path)
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    if with_dir:
      files = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
    return files


