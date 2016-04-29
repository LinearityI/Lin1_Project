#!/usr/bin/python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab


# Custom Module
import draw
from render import *
from common import *

def projectV(n,v): #projected vector
    n = n / np.linalg.norm(n)
    mag = np.dot(np.transpose(n),v)
    return mag * n

def projectP(n,v): #Projected plane
    return v - projectV(n,v) 

def projectedPoint(u, v, pp1, pp2):
    #finds closest point of "intersection" 
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


#origin
draw.point(vec(0,0,0))
draw.plane(vec(0,0,1),vec(0,0,0))


#near-plane
#draw.plane(vec(0,0,-1),vec(0,0,1))

#far-plane
#draw.plane(vec(0,0,4),vec(0,0,1))

#drawPlane(vec(0,0,1),vec(0,0,100))

#point

p = projectionMatrix(0.1,100,60*3.1415/180,1.0)
#p = projectionMatrix(1,10.0,20,20)
print 'p', p

draw.point(vec(0,0,-2),c=(0,0,0))

v = viewMatrix(
        vec(0,0,-2),#camera location
        vec(0,0,0),#object location
        vec(0,1,0) #'up' matrix
    )
print 'v', v

for i in np.linspace(-1,1,2):
    for j in np.linspace(-1,1,2):
        for k in np.linspace(-1,1,2):
            c = np.random.rand(3,1)
            c = (float(c[0]),float(c[1]),float(c[2]))

            pt = vec(i,j,k,1)
            draw.point(pt)
            ppt = np.dot(v,pt) #to camera-coordinates
            ppt = np.dot(p,ppt) #perspective projection

            ppt /= ppt[3] #z-divide
            print 'pt', pt
            print 'ppt', ppt
            draw.point(ppt,c=c)
            draw.line_pt(pt,ppt)

#for i in range(100):
#    n = -0.1
#    f = 10
#    pt = np.random.randn(4,1)*2
#    draw.point(pt)
#    ppt = np.dot(p,np.dot(v,pt))
#    draw.point(ppt,c=(0,0,1))
#    draw.line_pt(pt,ppt)

mlab.view(distance=10)
mlab.show()
