import os 
from os.path import isfile, join
from shutil import copyfile

def lsFiles(path="../dataset"):
    """
    Examples :
        >>> print(lsFiles())
        [file1.txt, file2.txt, file3.txt]

    Args:
        path (str, optional): The path of the directory we are interested in. Defaults  to "../dataset"

    Returns:
 	    Returns a list of strings where each element is the name of a file in the given path.

    """

    path = os.path.abspath(path)
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    return files
    #print files

def copyFile(fn, src_path, dst_path):
    """ 
    Args:
        fn (str): fn is the file we want to copy.
        src_path (str): src_path is the path of the source directory where the file exists.
        dst_path (str): dst_path is the path of the destination directory where I want to paste the file.

    """

    copyfile(join(src_path, fn), join(dst_path, fn))

    
def rmFile(fn, path):
    """
    Args:
        fn (str): fn is the file we want to delete.
        path (str): path of the directory where the file exists.
	
    """

    os.remove(join(path, fn))
