import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def vec(*args):
    """Returns arguments into column vector form"""
    return np.transpose(np.atleast_2d(args))

def projectV(n,v):
    """Returns the result of projection into a vector"""
    n = n / np.linalg.norm(n)
    mag = np.dot(np.transpose(n),v)
    return mag * n

def projectP(n,v):
    return v - projectV(n,v) 

def drawLine(fig,n,p):
    """Draws a line based on a normal vector & point"""
    s = np.linspace(-1,1)
    l = [p+e*n for e in s]

    xs = [e[0] for e in l]
    ys = [e[1] for e in l]
    zs = [e[2] for e in l]

    plt3d = fig.gca(projection='3d')
    plt3d.scatter(xs,ys,zs,'-',c='r')


def drawPlane(fig,n,p):
    d = -np.sum(p*n)
    x = np.linspace(-2,2)
    y = np.linspace(-2,2)
    [xx,yy]=np.meshgrid(x,y);
    z = (-n[0]*xx - n[1]*yy - d)/n[2]

    plt3d = fig.gca(projection='3d')
    plt3d.plot_surface(xx,yy,z)

def drawPoint(fig,p):
    plt3d = fig.gca(projection='3d')
    plt3d.scatter([p[0]],[p[1]],[p[2]],c='r',marker='o')

pt = np.random.rand(3,1)
print "pt", pt
v1 = np.random.rand(3,1)
#v1 = vec(0,0,1)
v2 = np.random.rand(3,1)

print "v1", v1
print "v2", v2

print "pp1", projectP(v1,pt)
print vec(0,0,0)



fig = plt.figure()
ax = fig.gca(projection='3d')
drawPlane(fig,v1,vec(0,0,0))
drawPlane(fig,v2,vec(0,0,0))



drawPoint(fig,pt)
pp1 = projectP(v1,pt)
pp2 = projectP(v2,pt)

drawPoint(fig,pp1)
drawPoint(fig,pp2)

drawLine(fig,v1,pt)
drawLine(fig,v2,pt)

def findIntersection(v1, pp1, v2, pp2):
    #Copied from DrawLine function, hopefully returns components
    s = np.linspace(-1,1)
    l = [p+e*n for e in s]

    xs = [e[0] for e in l]
    ys = [e[1] for e in l]
    zs = [e[2] for e in l] 




ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-5,5)


plt.show()