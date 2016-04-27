from mayavi import mlab
import numpy as np 

def vec(*args):
	return np.transpose(np.atleast_2d(args))

def viewMatrix(cam, obj, up):
	z = cam - obj#What the camera is looking at, z axis
	print 'z', z
	print 'up', up
	x = np.cross(z.T, up.T).T 
	y = np.cross(x.T, z.T).T

	tmp = np.hstack((x,y,z,vec(0,0,0)))
	print tmp
	
	cam = np.hstack((cam.T,[[1]]))
	print cam
	tmp = np.vstack((tmp,cam))
	return tmp

	#x = np.transpose(x)
	#y = np.transpose(y)
	#z = np.transpose(z)
	#cam = np.transpose(cam)

	#tmp = np.vstack((x, y, z, cam))
	#tmp = np.hstack((tmp,np.transpose([[0,0,0,1]])))
	#return tmp
def projectionMatrix(n,f,a,ar):
    """
    projectionMatrix(near,far,width,height)
    Calculates a projection of a frustrum unto a plane
    """
    n = float(n)
    f = float(f)
    #r = 0.5 * w
    #t = 0.5 * h
#    return np.asarray([
#                [n/r,0,0,0],
#                [0,n/t,0,0],
#                [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],
#                [0,0,1,0]
#            ])
    return np.asarray([
            [1/(ar*np.tan(a/2)), 0, 0, 0],
            [0, 1/np.tan(a/2), 0, 0],
            [0, 0, (-n-f)/(n-f), 2*f*n/(n-f)],
            [0, 0, 1, 0]
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
