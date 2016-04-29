import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab

def point(p,c=None):
    #p = p[:3]
    mlab.points3d(p[0],p[1],p[2],scale_factor=0.25,color=c,transparent=True,opacity=0.3)

def line(n,p): 
    s = np.linspace(-1,1)
    l = [p+e*n for e in s]

    xs = [e[0] for e in l]
    ys = [e[1] for e in l]
    zs = [e[2] for e in l]
    mlab.plot3d(xs,ys,zs)

def line_pt(p1,p2):
    p1 = p1[:3]
    p2 = p2[:3]

    s = np.linspace(0,1)
    v = p1-p2
    l = [p2 + e*v for e in s]

    xs = [e[0] for e in l]
    ys = [e[1] for e in l]
    zs = [e[2] for e in l]

    mlab.plot3d(xs,ys,zs)


def plane(n,p):
    d = -np.sum(p*n)
    x = np.linspace(-2,2)
    y = np.linspace(-2,2)
    [xx,yy]=np.meshgrid(x,y);
    zz = (-n[0]*xx - n[1]*yy - d)/n[2]

    mlab.mesh(xx,yy,zz,opacity=0.2)


