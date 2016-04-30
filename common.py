import numpy as np

def vec(*args):
    return np.transpose(np.atleast_2d(args))

def normalize(v):
    return v / np.linalg.norm(v)

def cross(v1,v2):
    return np.cross(v1.T,v2.T).T
def vdot(v1,v2):
    return [float(e) for e in np.dot(v1.T,v2)]
def rad(deg):
    return deg * 3.1415 / 180
