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

cam = vec(0,0,-1)
obj = vec(0,0,0)
objR = vec(1,0,0)

up = vec(0,1,0)

v = viewMatrix(cam,obj,up)
print v

objRP = np.vstack((objR,[[1]]))
print objRP
print np.dot(v,objRP)