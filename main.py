#!/usr/bin/python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab


# Custom Module
from draw import Painter
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
    a = np.sum(u*u)
    b = np.sum(u*v)
    c = np.sum(v*v)
    d = np.sum(u*w_0)
    e = np.sum(v*w_0)

    s_c = (b*e - c*d) / (a*c - b**2)
    t_c = (a*e - b*d) / (a*c - b**2)

    PP_1c = vec(*[float(e) for e in pp1 + s_c*u])
    PP_2c = vec(*[float(e) for e in pp2 + t_c*v])
    
    return (PP_1c+PP_2c)/2

def find_intersection(cam1, a1,b1, cam2, a2, b2):
    d1 = cam1.n
    a1 *= cam1.w/2 #cam1.w
    b1 *= cam1.h/2 #cam1.h


    cam_coord_transform1 = np.hstack((cam1.xaxis, cam1.yaxis, cam1.zaxis)) #T, matrix that helps you convert to the camera's coordinate system
    #cartesian_coord_transform1 = np.linalg.inv(cam_coord_transform1)
    newabd1 = np.dot(cam_coord_transform1, normalize(vec(a1, b1, d1))) #direction facing object from cam1
    newabd1 = vec(*[float(e) for e in newabd1])

    world.line(newabd1,cam1.pos,l=5)

    d2 = cam2.n
    a2 *= cam2.w/2 #cam2.w
    b2 *= cam2.h/2 #cam2.h

    cam_coord_transform2 = np.hstack((cam2.xaxis, cam2.yaxis, cam2.zaxis)) #T, matrix that helps you convert to the camera's coordinate system
    #cartesian_coord_transform2 = np.linalg.inv(cam_coord_transform2)
    newabd2 = np.dot(cam_coord_transform2, normalize(vec(a2, b2, d2))) #direction facing object from cam2
    newabd2 = vec(*[float(e) for e in newabd2])
    world.line(newabd2,cam2.pos,l=5,c=(0,0,0))

    return projectedPoint(newabd1, newabd2, cam1.pos, cam2.pos)


world = Painter(mlab.figure("World"))
proj1 = Painter(mlab.figure("Projection_1"))
proj2 = Painter(mlab.figure("Projection_2"))
reconstruct = Painter(mlab.figure("Reconstruct"))

#origin
world.point(vec(0,0,0))

#near-plane
#draw.plane(vec(0,0,-1),vec(0,0,1))

#far-plane
#draw.plane(vec(0,0,4),vec(0,0,1))

#drawPlane(vec(0,0,1),vec(0,0,100))

cam1 = Camera(0.5, 100, rad(90),1.0)

cam1.setpos(vec(-1.6,1.2,2.2))
cam1.lookat(vec(0,0,0))

world.point(cam1.pos, c=(0,0,0))
world.line(cam1.zaxis,cam1.pos,c=(0,0,1))
world.line(cam1.yaxis,cam1.pos,c=(0,1,0))
world.line(cam1.xaxis,cam1.pos,c=(1,0,0))

world.plane(cam1.zaxis,cam1.pos + cam1.n * cam1.zaxis)

cam2 = Camera(0.5, 100, rad(90),1.0)

cam2.setpos(vec(-2.2,1.6,-1.2))
world.point(cam2.pos, c=(0,0,0))
cam2.lookat(vec(0,0,0))

world.point(cam2.pos, c=(0,0,0))
world.line(cam2.zaxis,cam2.pos,c=(0,0,1))
world.line(cam2.yaxis,cam2.pos,c=(0,1,0))
world.line(cam2.xaxis,cam2.pos,c=(1,0,0))

world.plane(cam2.zaxis,cam2.pos + cam1.n * cam1.zaxis)

cols = [] #colors
pts = []
ppts1 = []
ppts2 = []


#draw cube
for i in np.linspace(-1,1,2):
    for j in np.linspace(-1,1,2):
        for k in np.linspace(-1,1,2):
            c = np.random.rand(3,1)
            c = (float(c[0]),float(c[1]),float(c[2]))
            cols += [c]


            #point
            pt = vec(i,j,k,1)
            pts += [pt]
            world.point(pt,c=c)
            
            #camera 1
            ppt1 = np.dot(cam1.v,pt) #to camera-coordinates
            ppt1 = np.dot(cam1.p,ppt1) #perspective projection

            ppt1 /= ppt1[3] #depth-divide
            ppts1 += [ppt1]
            #print 'ppt1', ppt1

            proj1.point(ppt1,c=c)

            #camera 2
            ppt2 = np.dot(cam2.v,pt) #to camera-coordinates
            ppt2 = np.dot(cam2.p,ppt2) #perspective projection

            ppt2 /= ppt2[3] #depth-divide
            ppts2 += [ppt2]

            proj2.point(ppt2,c=c)


for i in range(len(pts)):
    for j in range(len(pts)):
        if i>j and np.linalg.norm(pts[i] - pts[j]) <= 2:
            c = np.random.rand(3,1)
            c = (float(c[0]),float(c[1]),float(c[2]))
            world.line_pt(pts[i],pts[j],c=c)
            proj1.line_pt(ppts1[i],ppts1[j],c=c)
            proj2.line_pt(ppts2[i],ppts2[j],c=c)
            



#for i in range(100):
#    n = -0.1
#    f = 10
#    pt = np.random.randn(4,1)*2
#    draw.point(pt)
#    ppt = np.dot(p,np.dot(v,pt))
#    draw.point(ppt,c=(0,0,1))
#    draw.line_pt(pt,ppt)

for ppt1,ppt2,col in zip(ppts1,ppts2,cols):
    rpt = find_intersection(
        cam1,
        ppt1[0],
        ppt1[1],
        cam2,
        ppt2[0],
        ppt2[1]
    )
    print 'rpt', rpt
    reconstruct.point(rpt,c=col)

mlab.view(distance=10)
mlab.show()
