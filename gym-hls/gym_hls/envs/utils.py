import os  
from os.path import isfile, join
from shutil import copyfile

def lsFiles(path, with_dir=True):
    """ 
    Examples :
        >>> print(lsFiles(path, False))
        [file1.txt, file2.txt, file3.txt]

        >>> print(lsFiles(path))
        [path/file1.txt, path/file2.txt, path/file3.txt]

    Args:
        path (str): The path of the directory we are interested in.
        with_dir (bool, optional): with_dir should be set to True if you want the returned list (files) to
        contain the files’ names from the given directory (the parameter path) concatenated with the given
        path (the parameter path), or with_dir should be set to False if you want the returned list (files)
        to only contain the files’ names .Defaults to True.

    Returns:
 	    Returns a list of strings where each element is the name of a file in the given path (case when with_dir is False),
        or a list of strings where each element is the given path concatenated with a file name (case when with_dir is True).

    """
    path = os.path.abspath(path)
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    if with_dir:
      files = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
    return files


