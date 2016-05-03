import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab

class Painter:
    def __init__(self,fig):
        self.fig = fig

    def point(self,p,c=None):
        #p = p[:3]
        mlab.points3d(p[0],p[1],p[2],color=c,transparent=True,opacity=0.3,figure=self.fig,mode='point')

    def line(self,n,p,c=None,l=1,o=1,w=0.1): 
        s = np.linspace(0,l)
        l = [p+e*n for e in s]

        xs = [e[0] for e in l]
        ys = [e[1] for e in l]
        zs = [e[2] for e in l]
        mlab.plot3d(xs,ys,zs,figure=self.fig,color=c,opacity=o)

    def line_pt(self,p1,p2,c=None,w=0.01):
        p1 = p1[:3]
        p2 = p2[:3]

        s = np.linspace(0,1)
        v = p1-p2
        l = [p2 + e*v for e in s]

        xs = [e[0] for e in l]
        ys = [e[1] for e in l]
        zs = [e[2] for e in l]

        mlab.plot3d(xs,ys,zs,figure=self.fig,color=c,line_width=0.1,tube_radius=w)

    def plane(self,n,p,c=None):
        d = -np.sum(p*n)
        x = np.linspace(-2,2)
        y = np.linspace(-2,2)
        [xx,yy]=np.meshgrid(x,y);
        zz = (-n[0]*xx - n[1]*yy - d)/n[2]

        mlab.mesh(xx,yy,zz,opacity=0.15,figure=self.fig)

    def scene(self):
        return self.fig


