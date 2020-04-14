import numpy as np
def geomean(iterable):
    """
    Examples :
        >>> print(geomean([1, 3, 27]))
        4.32674871092
           
        >>> print(geomean([1,9,5,6,6,7])
        4.73989632394
	"""
    a = np.array(iterable).astype(float)
    prod = a.prod()
    prod = -prod if prod < 0 else prod
    return prod**(1.0/len(a))

