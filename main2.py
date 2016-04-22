import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab

def vec(*args):
    return np.transpose(np.atleast_2d(args))

def projectV(n,v):
    n = n / np.linalg.norm(n)
    mag = np.dot(np.transpose(n),v)
    return mag * n

def projectP(n,v):
    return v - projectV(n,v) 

def drawLine(n,p):
    s = np.linspace(-1,1)
    l = [p+e*n for e in s]

    xs = [e[0] for e in l]
    ys = [e[1] for e in l]
    zs = [e[2] for e in l]
    mlab.plot3d(xs,ys,zs)


def drawPlane(n,p):
    d = -np.sum(p*n)
    x = np.linspace(-2,2)
    y = np.linspace(-2,2)
    [xx,yy]=np.meshgrid(x,y);
    zz = (-n[0]*xx - n[1]*yy - d)/n[2]

    mlab.mesh(xx,yy,zz)

def drawPoint(p):
    mlab.points3d(p[0],p[1],p[2],scale_factor=.25)

pt = np.random.rand(3,1)
print "pt", pt
v1 = np.random.rand(3,1)
v2 = np.random.rand(3,1)

print "v1", v1
pp1 = projectP(v1,pt)
pp2 = projectP(v2,pt)

print "pp1", pp1 

drawPlane(v1,vec(0,0,0))
drawPlane(v2,vec(0,0,0))

drawPoint(pt)
drawPoint(pp1)
drawPoint(pp2)

drawLine(v1,pt)
drawLine(v2,pt)

#ax.set_xlim(-5,5)
#ax.set_ylim(-5,5)
#ax.set_zlim(-5,5)
mlab.view(distance=10)
mlab.show()
