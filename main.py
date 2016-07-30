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
    
    return (PP_1c+PP_2c)/2.

def find_intersection(cam1, a1,b1, cam2, a2, b2, cam3, a3, b3):
    global world
    cam_coord_transform1 = np.hstack((cam1.xaxis, cam1.yaxis, cam1.zaxis)) #T, matrix that helps you convert to the camera's coordinate system
    

    a1 *= cam1.w
    b1 *= cam1.h

    n1 = cross(vec(a1, b1, 0),vec(0,0,1)) #in camera's coordinates
    n1 = np.dot(cam_coord_transform1,n1)
    n1 = vec(*[float(e) for e in n1])
    n1 = normalize(n1)
    p1 = cam1.pos

    #world.plane(n1,p1)
    
    cam_coord_transform2 = np.hstack((cam2.xaxis, cam2.yaxis, cam2.zaxis)) #T, matrix that helps you convert to the camera's coordinate system
    
    a2 *= cam2.w
    b2 *= cam2.h
    
    n2 = cross(vec(a2, b2, 0),vec(0,0,1)) #in camera's coordinates
    n2 = np.dot(cam_coord_transform2,n2)
    n2 = vec(*[float(e) for e in n2])
    n2 = normalize(n2)
    p2 = cam2.pos

    #world.plane(n2,p2)
    
    cam_coord_transform3 = np.hstack((cam3.xaxis, cam3.yaxis, cam3.zaxis)) #T, matrix that helps you convert to the camera's coordinate system
    

    a3 *= cam3.w
    b3 *= cam3.h
    
    n3 = cross(vec(a3, b3, 0),vec(0,0,1)) #in camera's coordinates
    n3 = np.dot(cam_coord_transform3,n3)
    n3 = vec(*[float(e) for e in n3])
    n3 = normalize(n3)
    p3 = cam3.pos

    #world.plane(n3,p3)
    
    n1n2n3_matrix = np.hstack((n1, n2, n3))
    #print 'n1n2n3', n1n2n3_matrix

    finalpt = (
        vdot(p1,n1)*cross(n2, n3)
    + vdot(p2,n2)*cross(n3, n1)
    + vdot(p3,n3)*cross(n1, n2)
    )/(np.linalg.det(n1n2n3_matrix))

    return finalpt


world = Painter(mlab.figure("World"))

proj1 = Painter(mlab.figure("Projection_1"))
proj2 = Painter(mlab.figure("Projection_2"))
proj3 = Painter(mlab.figure("Projection_3"))

reconstruct = Painter(mlab.figure("Reconstruct"))

#origin
world.point(vec(0,0,0))

#near-plane
#draw.plane(vec(0,0,-1),vec(0,0,1))

#far-plane
#draw.plane(vec(0,0,4),vec(0,0,1))

#drawPlane(vec(0,0,1),vec(0,0,100))

cam1 = Camera(0.3, 100, rad(60),1.5)

cam1.setpos(vec(-1.6, -2.6, -1.6))
cam1.lookat(vec(0,0,0.5))

world.point(cam1.pos, c=(0,0,0))
world.line(cam1.zaxis,cam1.pos,c=(0,0,1))
world.line(cam1.yaxis,cam1.pos,c=(0,1,0))
world.line(cam1.xaxis,cam1.pos,c=(1,0,0))

#draw frustrum borders
world.line(cam1.zaxis + cam1.yaxis / np.tan(cam1.fov/2), cam1.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam1.zaxis - cam1.yaxis / np.tan(cam1.fov/2), cam1.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam1.zaxis + cam1.xaxis / np.tan(cam1.fov/2), cam1.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam1.zaxis - cam1.xaxis / np.tan(cam1.fov/2), cam1.pos, c=(1,1,1),o=0.2,l=3)

#world.plane(cam1.zaxis,cam1.pos + cam1.n * cam1.zaxis)

cam2 = Camera(0.5, 100, rad(60),1)

cam2.setpos(vec(4.4, -5.2, -1.51))
world.point(cam2.pos, c=(0,0,0))
cam2.lookat(vec(1.0,0,0))

world.point(cam2.pos, c=(0,0,0))
world.line(cam2.zaxis,cam2.pos,c=(0,0,1))
world.line(cam2.yaxis,cam2.pos,c=(0,1,0))
world.line(cam2.xaxis,cam2.pos,c=(1,0,0))

#draw frustrum borders
world.line(cam2.zaxis + cam2.yaxis / np.tan(cam2.fov/2) + cam2.xaxis / np.tan(cam2.fov/2), cam2.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam2.zaxis + cam2.yaxis / np.tan(cam2.fov/2) - cam2.xaxis / (cam2.ar * np.tan(cam2.fov/2)), cam2.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam2.zaxis - cam2.yaxis / np.tan(cam2.fov/2) + cam2.xaxis / np.tan(cam2.fov/2), cam2.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam2.zaxis - cam2.yaxis / np.tan(cam2.fov/2) - cam2.xaxis / (cam2.ar * np.tan(cam2.fov/2)), cam2.pos, c=(1,1,1),o=0.2,l=3)
#world.plane(cam2.zaxis,cam2.pos + cam1.n * cam1.zaxis)

cam3 = Camera(0.5, 100, rad(60),1.0)

cam3.setpos(vec(-3,4.2, -5.4))
world.point(cam3.pos, c=(0,0,0))
cam3.lookat(vec(0,0.11,0))

world.point(cam3.pos, c=(0,0,0))
world.line(cam3.zaxis,cam3.pos,c=(0,0,1))
world.line(cam3.yaxis,cam3.pos,c=(0,1,0))
world.line(cam3.xaxis,cam3.pos,c=(1,0,0))

world.line(cam3.zaxis + cam3.yaxis / np.tan(cam3.fov/2), cam3.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam3.zaxis - cam3.yaxis / np.tan(cam3.fov/2), cam3.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam3.zaxis + cam3.xaxis / np.tan(cam3.fov/2), cam3.pos, c=(1,1,1),o=0.2,l=3)
world.line(cam3.zaxis - cam3.xaxis / np.tan(cam3.fov/2), cam3.pos, c=(1,1,1),o=0.2,l=3)

cols = [] #colors
pts = []
ppts1 = []
ppts2 = []
ppts3 = []

connect = np.zeros((800,800),dtype=bool)

for line in open("teapot_4.obj", "r"):
    if line.startswith('#'):
        continue
    values = line.split()
    if not values:
        continue
    if values[0] == 'v': #vertex
        v = map(float, values[1:4])
        pts += [vec(v[0],v[1],v[2],1.0)]
    if values[0] == 'f':
        v = map(int, values[1:4])
        for i in range(3):
            for j in range(3):
                if v[i] > v[j]:
                    connect[v[i]][v[j]] = True

#draw cube
#for i in np.linspace(-1,1,2):
#    for j in np.linspace(-1,1,2):
#        for k in np.linspace(-1,1,2):
#            pt = vec(i,j,k,1)
#            pts += [pt]

i = 0
for pt in pts:
    print i
    i += 1
    c = np.random.rand(3,1)
    c = (float(c[0]),float(c[1]),float(c[2]))
    cols += [c]

    world.point(pt,c=c)
    
    #camera 1
    ppt1 = np.dot(cam1.v,pt) #to camera-coordinates
    ppt1 = np.dot(cam1.p,ppt1) #perspective projection

    ppt1 /= ppt1[3] #perspective-divide
    ppts1 += [ppt1]

    proj1.point(ppt1,c=c)

    #camera 2
    ppt2 = np.dot(cam2.v,pt) #to camera-coordinates
    ppt2 = np.dot(cam2.p,ppt2) #perspective projection

    ppt2 /= ppt2[3] #perspective-divide
    ppts2 += [ppt2]

    proj2.point(ppt2,c=c)


    #camera 3
    ppt3 = np.dot(cam3.v,pt) #to camera-coordinates
    ppt3 = np.dot(cam3.p,ppt3) #perspective projection

    ppt3 /= ppt2[3] #perspective-divide
    ppts3 += [ppt3]

    proj3.point(ppt3,c=c)

rpts = []

i = 0

for ppt1,ppt2,ppt3,col in zip(ppts1,ppts2,ppts3,cols):
    print i
    i += 1
    rpt = find_intersection(
        cam1,
        ppt1[0],
        ppt1[1],
        cam2,
        ppt2[0],
        ppt2[1],
        cam3,
        ppt3[0],
        ppt3[1]
    )
    rpts += [rpt]
    reconstruct.point(rpt,c=col)

k = 0
c = np.random.rand(3,1)
c = (float(c[0]),float(c[1]),float(c[2]))

for i in range(len(pts)):
    for j in range(len(pts)):
        print k
        k += 1
        if connect[i][j]:
        #if i>j and np.random.random_sample() < 0.1:
            #c = np.random.rand(3,1)
            #c = (float(c[0]),float(c[1]),float(c[2]))
            world.line_pt(pts[i],pts[j],c=c,w=0.03)
            proj1.line_pt(ppts1[i],ppts1[j],c=c,w=0.005)
            proj2.line_pt(ppts2[i],ppts2[j],c=c,w=0.001)
            proj3.line_pt(ppts3[i],ppts3[j],c=c,w=0.01)
            reconstruct.line_pt(rpts[i],rpts[j],c=c)
            



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
