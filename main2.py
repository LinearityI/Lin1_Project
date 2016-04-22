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

def projectedPoint(u, v, pp1, pp2):
	#normalizing the vectors
	#u = u/np.linalg.norm(u)
	#v = u/np.linalg.norm(v)
	w_0 = pp1 - pp2 #vector from pp1 to pp2
	a = np.dot(np.transpose(u),u) #supposed to be 1?
	b = np.dot(np.transpose(u),v)
	c = np.dot(np.transpose(v),v) #supposed to be 1?
	d = np.dot(np.transpose(u),w_0)
	e = np.dot(np.transpose(v),w_0)

	s_c = (b*e - c*d) / (a*c - b**2)
	t_c = (a*e - b*d) / (a*c - b**2)

	PP_1c = pp1 + s_c*u
	PP_2c = pp2 + t_c*v

	return PP_1c, PP_2c





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

mlab.view(distance=10)
mlab.show()
