import numpy as np

def vec(*args):
    return np.transpose(np.atleast_2d(args))

def normalize(v):
    return v / np.linalg.norm(v)

def cross(v1,v2):
	return np.cross(v1.T,v2.T).T