from common import *
from mayavi import mlab
import numpy as np 

def viewMatrix(cam, obj, up):

        f = normalize(cam - obj)
        up = normalize(up)
        s = np.cross(f[:,0],up[:,0])
        up = np.cross(s,f[:,0])
        print 's', s
        print 'up', up
        #up = np.cross(s.T,f.T).T


        #rotation
        M1 = np.asarray([
            [s[0],s[1],s[2],0],
            [up[0],up[1],up[2],0],
            [f[0],f[1],f[2],0],
            [0,0,0,1]
            ])
        print 'M1', M1
        
        #translation
        M2 = np.asarray([
            [1,0,0,-cam[0]],
            [0,1,0,-cam[1]],
            [0,0,1,-cam[2]],
            [0,0,0,1]
            ])

        print 'M2', M2
        return np.dot(M1,M2)


#	z = normalize(cam - obj)#What the camera is looking at, z axis
#        up = normalize(up)
#
#	print 'z', z
#	print 'up', up
#	x = np.cross(up.T, z.T).T 
#	y = np.cross(z.T, x.T).T
#        print 'x', x
#        print 'y', y
#
#	tmp = np.hstack((x,y,z,vec(0,0,0)))
#        translate = vec(
#                float(np.dot(x.T,cam)),
#                float(np.dot(y.T,cam)),
#                float(np.dot(z.T,cam))
#                )
#        print 'TTT', translate
#        cam = np.hstack((-translate.T,[[1]]))
#
#        return np.vstack((tmp,cam))
#
	#print tmp
	#
	#cam = np.hstack((cam.T,[[1]]))
	#print cam
	#tmp = np.vstack((tmp,cam))
	#return tmp

	#x = np.transpose(x)
	#y = np.transpose(y)
	#z = np.transpose(z)
	#cam = np.transpose(cam)

	#tmp = np.vstack((x, y, z, cam))
	#tmp = np.hstack((tmp,np.transpose([[0,0,0,1]])))
	#return tmp

def projectionMatrix(n,f,fov,ar):
    """
    projectionMatrix(near,far,width,height)
    Calculates a projection of a frustrum unto a plane
    """

    n = float(n)
    f = float(f)

    fov = float(fov)
    ar = float(ar)
    print 'ar', ar

    #r = 0.5 * w
    #t = 0.5 * h
    #perspective, w-h
    #return np.asarray([
    #            [n/r,0,0,0],
    #            [0,n/t,0,0],
    #            [0,0,(f+n)/(f-n),-2*f*n/(f-n)],
    #            [0,0,1,0]
    #        ])
    #orthographic
#    return np.asarray([
#                [1./r,0,0,0],
#                [0,1./t,0,0],
#                [0,0,-2./(f-n),-(f+n)/(f-n)],
#                [0,0,0,1]
#            ])
    #perspective, fov-aspect
    return np.asarray([
            [1/(ar*np.tan(fov/2)), 0, 0, 0],
            [0, 1/np.tan(fov/2), 0, 0],
            [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)],
            [0, 0, -1, 0]
        ])

#cam = vec(0,0,-1)
#obj = vec(0,0,0)
#objR = vec(1,0,0)
#
#up = vec(0,1,0)
#
#v = viewMatrix(cam,obj,up)
#print v
#
#objRP = np.vstack((objR,[[1]]))
#print objRP
#print np.dot(v,objRP)


#obj = vec(1,1,1,5)
#p = projectionMatrix(0.1,100.0,10.0,10.0)
#print np.dot(p,obj)
