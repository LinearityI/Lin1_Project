#!/usr/bin/python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab


# Custom Module
from rendering import *


def vec(*args):
    return np.transpose(np.atleast_2d(args))

def projectV(n,v): #projected vector
    n = n / np.linalg.norm(n)
    mag = np.dot(np.transpose(n),v)
    return mag * n

def projectP(n,v): #Projected plane
    return v - projectV(n,v) 

def drawLine(n,p): 
    s = np.linspace(-1,1)
    l = [p+e*n for e in s]

    xs = [e[0] for e in l]
    ys = [e[1] for e in l]
    zs = [e[2] for e in l]
    mlab.plot3d(xs,ys,zs)

def drawLine_pt(p1,p2):
    p1 = p1[:3]
    p2 = p2[:3]

    s = np.linspace(0,1)
    v = p1-p2
    l = [p2 + e*v for e in s]

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

def drawPoint(p,c=None):
    p = p[:3]
    mlab.points3d(p[0],p[1],p[2],scale_factor=.25,color=c)

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


#
#
#
#pt = np.random.rand(3,1)
#print "pt-original", pt
#v1 = np.random.rand(3,1)
#v2 = np.random.rand(3,1)
#
#print "v1", v1
#pp1 = projectP(v1,pt)
#pp2 = projectP(v2,pt)
#
#print "pp1", pp1 
#
#drawPlane(v1,vec(0,0,0))
#drawPlane(v2,vec(0,0,0))
#
#drawPoint(pt)
#drawPoint(pp1)
#drawPoint(pp2)
#
#drawLine(v1,pt)
#drawLine(v2,pt)
#
#print 'pt-reconstructed', projectedPoint(v1, v2, pp1, pp2)
#
#mlab.view(distance=10)
#mlab.show()


#origin
drawPoint(vec(0,0,0))
drawPlane(vec(0,0,1),vec(0,0,0))


#near-plane
drawPlane(vec(0,0,-1),vec(0,0,1))

#far-plane
drawPlane(vec(0,0,4),vec(0,0,1))

#drawPlane(vec(0,0,1),vec(0,0,100))

#point

p = projectionMatrix(1,5,60*3.14/180,1.0)
print p
D,V = np.linalg.eig(p)
print 'Inverse', np.linalg.inv(p)
print 'V', V
print 'D', D

print p

pts = []

for i in [-1,1]:
    for j in [-1,1]:
        for k in [-1,1]:
            print i,j,k
            pt = vec(i,j,k+2,1)
            pts += [pt]
            drawPoint(pt)
            drawLine_pt(pt,vec(0,0,0))
            ppt = np.dot(p,pt)
            print 'ppt', ppt
            drawPoint(ppt,c=(1,0,0))


#for pt1 in pts:
#    for pt2 in pts:
#        drawLine_pt(pt1,pt2)


pt = vec(2,4,3,1)
drawPoint(vec(2,4,3))

#projected point
ppt = np.dot(p,pt)

print ppt

drawPoint(ppt,c=(0,1,0))

drawLine_pt(pt,vec(0,0,0))

mlab.view(distance=10)
mlab.show()
