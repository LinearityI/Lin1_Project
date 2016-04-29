import numpy as np

def vec(*args):
    return np.transpose(np.atleast_2d(args))

def normalize(v):
    return v / np.linalg.norm(v)
