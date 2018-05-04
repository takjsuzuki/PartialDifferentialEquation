import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d


class WaveEquation():

    def __init__(self): 
        self.Lx = 41
        self.Ly = 41
        self.dx = 1e-2
        self.dy = 1e-2
        self.dx = 1.0/self.Lx
        self.dy = 1.0/self.Ly
        self.dt=(1/(1/self.dx+1/self.dy))*0.5
        self.u = np.zeros((self.Lx,self.Ly),dtype=float)
        self.v = np.zeros((self.Lx,self.Ly),dtype=float)
        self.c = 1.0
        self.time = 0.0
        
        self.circle = np.zeros((self.Lx,self.Ly),dtype=float)

        
    def Initialize(self):

        sigma = self.dx*2
        Gauss = lambda x: np.exp(-x**2.0/(2.0*sigma**2))/(np.sqrt(2.0*np.pi*sigma**2))
        
        x, y = np.meshgrid((np.arange(self.Lx)-self.Lx/2)*self.dx,
                           (np.arange(self.Ly)-self.Ly/2)*self.dy)
        self.u = Gauss(x)*Gauss(y)/(Gauss(0.0)**2.0)
        
        self.v[1:-1,1:-1]\
            = self.u[1:-1,1:-1] \
            +0.5*((self.c*self.dt/self.dx)**2)\
            *(self.u[2:,1:-1]+self.u[:-2,1:-1]-2.0*self.u[1:-1,1:-1])\
              +0.5*((self.c*self.dt/self.dy)**2)*(self.u[1:-1,2:]+self.u[1:-1,:-2]
                                                  -2.0*self.u[1:-1,1:-1])

        x, y = np.meshgrid((np.arange(self.Lx)-self.Lx/2),(np.arange(self.Ly)-self.Ly/2))
        self.circle[(x**2+y**2)<(self.Lx/2)**2] = 1.0
        
            
    def Forward_step(self):

        self.BoundaryCondition()
        
        u0 = self.u.copy()
        self.u = self.v.copy()

        self.v[1:-1,1:-1]\
            = 2.0*self.u[1:-1,1:-1] - u0[1:-1,1:-1]  \
            +((self.c*self.dt/self.dx)**2)*(self.u[2:,1:-1]+self.u[:-2,1:-1]
                                            -2.0*self.u[1:-1,1:-1])\
            +((self.c*self.dt/self.dy)**2)*(self.u[1:-1,2:]+self.u[1:-1,:-2]
                                            -2.0*self.u[1:-1,1:-1])

        self.time += self.dt

    
    def Forward(self,n):

        for i in range(n):
            # print("{0}th step done".format(i))
            self.Forward_step()
        
    def BoundaryCondition(self):
        
        # self.u[:,0]  = 0.0
        # self.u[:,-1] = 0.0
        # self.u[0,:]  = 0.0
        # self.u[-1,:] = 0.0

        # self.v[:,0]  = 0.0
        # self.v[:,-1] = 0.0
        # self.v[0,:]  = 0.0
        # self.v[-1,:] = 0.0

        self.u *= self.circle
        self.v *= self.circle
        
    def ReturnValue(self):
        return self.u

    def Output(self,Mode):

        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        plt.subplots_adjust(left=0.0, bottom=0.05, right=0.95, top=1.0)

        NFrames = 100
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        def plot(i):
            plt.cla()                      

            ax.text2D(0.05, 0.90,
                      r"$\frac{1}{c^2} \frac{\partial^2 u}{\partial t^2} = \left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2} \right)u$",
                  transform=ax.transAxes,size=20)
            ax.text2D(0.05, 0.78,
                      r"$t=$"+str("{:.2f}".format(round(self.time,2))),
                      transform=ax.transAxes,size=20)

            ax.set_xlabel(r'$x$',size=20)
            ax.set_ylabel(r'$y$',size=20)
            ax.set_zlabel(r'$u$',size=20)


            p = Circle((self.Lx/2,self.Ly/2), self.Lx/2, facecolor='none',
                       edgecolor="black", linewidth=2.0, alpha=0.6)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
            
            z = self.u.copy()
            z[self.circle==0.0]= np.nan
            
            ax.set_zlim(-0.5,1.0)
            ax.plot_wireframe(x,y,z, color='r',rstride=2,cstride=2)
            
            self.Forward(2)
        
            print("{0} / {1} calculated".format(i+1,NFrames))


        
        if Mode == "Display":        

            for i in range(NFrames):
                plot(i)
                plt.draw()
                plt.pause(0.001)            


        elif Mode == "MakeGif":
            ani = animation.FuncAnimation(fig, plot, frames=NFrames, interval=50, blit=False)
            ani.save("output.gif", writer="imagemagick")        
