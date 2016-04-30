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
    a = float(np.dot(u.T,u)) #supposed to be 1?
    b = float(np.dot(u.T,v))
    c = float(np.dot(v.T,v)) #supposed to be 1?
    d = float(np.dot(u.T,w_0))
    e = float(np.dot(v.T,w_0))

    s_c = (b*e - c*d) / (a*c - b**2)
    t_c = (a*e - b*d) / (a*c - b**2)

    PP_1c = vec(*[float(e) for e in pp1 + s_c*u])
    PP_2c = vec(*[float(e) for e in pp2 + t_c*v])
    
    return (PP_1c+PP_2c)/2


world = Painter(mlab.figure("World"))
proj = Painter(mlab.figure("Projection"))
reconstruct = Painter(mlab.figure("Reconstruct"))

#origin
world.point(vec(0,0,0))
world.plane(vec(0,0,1),vec(0,0,0))


#near-plane
#draw.plane(vec(0,0,-1),vec(0,0,1))

#far-plane
#draw.plane(vec(0,0,4),vec(0,0,1))

#drawPlane(vec(0,0,1),vec(0,0,100))

#point
near = 0.1
far = 1000
fov = 45*3.1415/180 #field of view, converted to radians
aspect_ratio = 1.0
p = projectionMatrix(near,far,fov,aspect_ratio)
#p = projectionMatrix(1,10.0,20,20)
print 'p', p

camera1_location = vec(0,-4,-2)
#camera 1
world.point(vec(0,-4,-2),c=(0,0,0))
camera_direction1, up1, side1, final_viewMatrix1 = viewMatrix(
        camera1_location,#camera location
        vec(0,0,0),#object location
        vec(0,1,0) #'up' matrix
    )

camera2_location = vec(-1,2,3)
#camera 2
world.point(vec(0,-4,-2),c=(0,0,0))
camera_direction2, up2, side2, final_viewMatrix2 = viewMatrix(
        camera2_location,#camera location
        vec(0,0,0),#object location
        vec(0,1,0) #'up' matrix
    )


pts = []
ppts1 = []
ppts2 = []


for i in np.linspace(-1,1,2):
    for j in np.linspace(-1,1,2):
        for k in np.linspace(-1,1,2):
            c = np.random.rand(3,1)
            c = (float(c[0]),float(c[1]),float(c[2]))

            pt = vec(i,j,k,1)
            pts += [pt]
            world.point(pt)
            
            ppt1 = np.dot(final_viewMatrix1,pt) #to camera-coordinates
            ppt1 = np.dot(p,ppt1) #perspective projection

            ppt1 /= ppt1[3] #depth-divide
            ppts1 += [ppt1]
            #print 'pt', pt
            # print 'ppt', ppt

            ppt2 = np.dot(final_viewMatrix2,pt) #to camera-coordinates
            ppt2 = np.dot(p,ppt2) #perspective projection

            ppt2 /= ppt2[3] #depth-divide
            ppts2 += [ppt2]
            #print 'pt', pt
            # print 'ppt', ppt
#print ppts1, ppts2


for i in range(len(pts)):
    for j in range(len(pts)):
        if i>j and np.linalg.norm(pts[i] - pts[j]) <= 2:
            c = np.random.rand(3,1)
            c = (float(c[0]),float(c[1]),float(c[2]))
            world.line_pt(pts[i],pts[j],c=c)
            proj.line_pt(ppts1[i],ppts1[j],c=c)
            



#for i in range(100):
#    n = -0.1
#    f = 10
#    pt = np.random.randn(4,1)*2
#    draw.point(pt)
#    ppt = np.dot(p,np.dot(v,pt))
#    draw.point(ppt,c=(0,0,1))
#    draw.line_pt(pt,ppt)

def find_intersection(cam1, cam2, dir1, dir2, a1, a2, b1, b2):
    d1 = near
    up1 = vec(0, 1.0, 0) #basically the y axis

    cam1x = side1
    cam1y = up1
    cam1z = cross(cam1x, cam1y)

    print 'cam1x', cam1x
    print 'cam1y', cam1y
    print 'cam1z', cam1z

    cam_coord_transform1 = np.hstack((cam1x, cam1y, cam1z)) #T, matrix that helps you convert to the camera's coordinate system
    print 'cct', cam_coord_transform1

    cartesian_coord_transform1 = np.linalg.inv(cam_coord_transform1)
    print 'cartcoordtrans', cartesian_coord_transform1

    newabd1 = np.dot(cartesian_coord_transform1, vec(a1, b1, d1)) #direction facing object from cam1
    print 'newabd1', newabd1

    d2 = near    
    up2 = vec(0, 1.0, 0) #basically the y axis
    cam2x = side2
    cam2y = up2
    cam2z = cross(cam2x, cam2y)
    cam_coord_transform2 = np.hstack((cam2x, cam2y, cam2z)) #T, matrix that helps you convert to the camera's coordinate system
    cartesian_coord_transform2 = np.linalg.inv(cam_coord_transform2)
    newabd2 = np.dot(cartesian_coord_transform2, vec(a2, b2, d2)) #direction facing object from cam2

    return projectedPoint(newabd1, newabd2, cam1, cam2)


for ppt1,ppt2 in zip(ppts1,ppts2):
    print ppt1[0]
    rpt = find_intersection(
        camera1_location,
        camera2_location,
        camera_direction1,
        camera_direction2,
        ppt1[0],
        ppt2[0],
        ppt1[1],
        ppt2[1]
    )
    print 'rpt', rpt
    reconstruct.point(rpt,c=(0,0,0))

mlab.view(distance=10)
mlab.show()
