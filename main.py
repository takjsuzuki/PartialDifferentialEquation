import numpy as np
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Circle
import matplotlib.animation as animation
import matplotlib.cm as cm

from matplotlib.animation import ArtistAnimation

def animate(i):
    plt.cla()
    HD = HeatDiffusion()
    HD.Display()

fig = plt.figure()
# ax = Axes3D(fig)
# ax = fig.add_subplot(111)
ax = fig.add_subplot(111,projection="3d")


plt.subplots_adjust(left=0.0, bottom=0.05, right=0.95, top=1.0)


class HeatDiffusion():

    def __init__(self):
        self.Lx = 200
        self.Ly = 200
        self.dt = 1e-5
        self.dx = 1e-2
        self.dy = 1e-2
        self.u = np.zeros((self.Lx,self.Ly),dtype=float)
        self.D = 1.0
        self.time = 0.0
        
    def Initialize(self):

        sigma = (self.dx*self.Lx)*0.05
        Gauss = lambda x: np.exp(-(x**2.0/(2.0*sigma**2)))/(np.sqrt(2.0*np.pi*sigma**2))
        
        
        x, y = np.meshgrid((np.arange(self.Lx)-self.Lx/2)*self.dx,
                           (np.arange(self.Ly)-self.Ly/2)*self.dy)
        self.u = Gauss(x)*Gauss(y)

        for i in range(self.Lx):
            for j in range(self.Ly):
                self.u[i,j] = 1.0 if ((np.abs(i-self.Lx/2)<20) and (np.abs(j-self.Ly/2)<20)) else 0.0
        
        
                
    def Forward_step(self):

        ui = self.u.copy()

        self.BoundaryCondition()
        
        self.u[1:-1, 1:-1] \
        += self.dt*self.D*((ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/(self.dx**2)
            + (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/(self.dy**2)
        )


    def Forward(self,n):

        for i in range(n):
            print("{0}th step done".format(i))
            self.Forward_step()
        
    def BoundaryCondition(self):
        
        self.u[:,0]  = 0.0
        self.u[:,-1] = 0.0
        self.u[0,:]  = 0.0
        self.u[-1,:] = 0.0
        
    def ReturnValue(self):
        return self.u

    def plot(self,data):
        plt.cla()                      
        self.Forward(500)
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        ax.plot_wireframe(x,y,self.u, color='r',rstride=10,cstride=10)
        
    def Animation(self):

        
        #fig = plt.figure()
        # ax = Axes3D(fig)
        # x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        # ax.set_xlabel('z')
        # ax.set_ylabel('y')
        # ax.set_zlabel('u')
        # anim = []
        # for i in range(10):
        #     HD.Forward(10)
        #     im = ax.plot_wireframe(x,y,self.u, color='r')
        #     anim.append(im)

        # anim = ArtistAnimation(fig, anim)

        # ani = animation.FuncAnimation(fig, animate,
        #                               interval = 1, frames = 2)


        ani = animation.FuncAnimation(fig, self.plot, frames=20, interval=50, blit=False)

        ani.save("output.gif", writer="imagemagick")
        
        # anim = animation.FuncAnimation(fig, self.Forward, 10, fargs=(),
        #                                interval=10, blit=False)
        #anim.save('HeatDiffusion.gif', writer=writer)        
        #plt.show() 
    
        # anim.save("t.gif", writer='imagemagick')
        

    
    def Display(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        ax.plot_wireframe(x,y,self.u, color='r')
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        plt.show()




class WaveEquation():

    def __init__(self): 
        # self.Lx = 200
        # self.Ly = 200
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
        
        self.v[1:-1,1:-1]= self.u[1:-1,1:-1]\
                +0.5*((self.c*self.dt/self.dx)**2)*(self.u[2:,1:-1]+self.u[:-2,1:-1]-2.0*self.u[1:-1,1:-1])\
                +0.5*((self.c*self.dt/self.dy)**2)*(self.u[1:-1,2:]+self.u[1:-1,:-2]-2.0*self.u[1:-1,1:-1])


        x, y = np.meshgrid((np.arange(self.Lx)-self.Lx/2),
                           (np.arange(self.Ly)-self.Ly/2))
        self.circle[(x**2+y**2)<(self.Lx/2)**2] = 1.0
        
            
    def Forward_step(self):

        self.BoundaryCondition()
        
        u0 = self.u.copy()
        self.u = self.v.copy()

        self.v[1:-1,1:-1] = 2.0*self.u[1:-1,1:-1] - u0[1:-1,1:-1] \
                 +((self.c*self.dt/self.dx)**2)*(self.u[2:,1:-1]+self.u[:-2,1:-1]-2.0*self.u[1:-1,1:-1])\
                 +((self.c*self.dt/self.dy)**2)*(self.u[1:-1,2:]+self.u[1:-1,:-2]-2.0*self.u[1:-1,1:-1])

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

    def plot(self,data):
        plt.cla()                      

        # ax.set_title("Wave Equation")
        
        ax.text2D(0.05, 0.90,
                  r"$\frac{1}{c^2} \frac{\partial^2 u}{\partial t^2} = \left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2} \right)u$",
                  transform=ax.transAxes,size=20)
        # ax.text2D(0.05, 0.79,"Boundary: disk",transform=ax.transAxes,size=20)
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

        # pathpatch_2d_to_3d(p, z = 0, normal = 'z')
        # pathpatch_translate(p, (0.5, 0.5, 0))

        
        
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))

        z = self.u.copy()
        z[self.circle==0.0]= np.nan
        
        ax.set_zlim(-0.5,1.0)
        #ax.plot_wireframe(x,y,self.u, color='r',rstride=10,cstride=10)
        ax.plot_wireframe(x,y,z, color='r',rstride=2,cstride=2)

        self.Forward(2)
        
        
    def Animation(self):

        #ani = animation.FuncAnimation(fig, self.plot, frames=200, interval=50, blit=False)
        ani = animation.FuncAnimation(fig, self.plot, frames=200, interval=10, blit=False)
        #ani = animation.FuncAnimation(fig, self.plot, frames=10, interval=10, blit=False)
        ani.save("output.gif", writer="imagemagick")        

    
    def Display(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        ax.plot_wireframe(x,y,self.u, color='r')
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        plt.show()

        




class ReactionDiffusion():

    def __init__(self):
        self.Lx = 200
        self.Ly = 200
        self.dt = 1e-5
        self.dx = 1e-2
        self.dy = 1e-2
        self.u = np.zeros((self.Lx,self.Ly),dtype=float)
        self.v = np.zeros((self.Lx,self.Ly),dtype=float)
        self.a= 1.0
        self.b= 1.0
        self.k= -1.0
        self.tau= 1.0

        # self.a = 2.8e-4
        # self.b = 5e-3
        # self.tau = .1
        # self.k = -.005
        
    def Initialize(self):

        # sigma = (self.dx*self.Lx)*0.05
        # Gauss = lambda x: np.exp(-(x**2.0/(2.0*sigma**2)))/(np.sqrt(2.0*np.pi*sigma**2))
        
        
        # x, y = np.meshgrid((np.arange(self.Lx)-self.Lx/2)*self.dx,
        #                    (np.arange(self.Ly)-self.Ly/2)*self.dy)
        # self.u = Gauss(x)*Gauss(y)

        # for i in range(self.Lx):
        #     for j in range(self.Ly):
        #         self.u[i,j] = 1.0 if ((np.abs(i-self.Lx/2)<20) and (np.abs(j-self.Ly/2)<20)) else 0.0

        self.u = np.random.randn(self.Lx,self.Ly)
        self.v = np.random.randn(self.Lx,self.Ly)
        
                
    def Forward_step(self):

        ui = self.u.copy()
        vi = self.v.copy()

        self.BoundaryCondition()
        
        self.u[1:-1, 1:-1] \
        += self.dt*self.a*((ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/(self.dx**2)
                           +(ui[1:-1,2:] - 2*ui[1:-1, 1:-1] + ui[1:-1,:-2])/(self.dy**2)
                           +ui[1:-1, 1:-1]-ui[1:-1, 1:-1]**3.0
                           - vi[1:-1, 1:-1] +self.k)

        self.v[1:-1, 1:-1] \
        += self.dt/self.tau*((vi[2:, 1:-1]-2*vi[1:-1, 1:-1]+vi[:-2, 1:-1])*self.b/(self.dx**2)
                             +(vi[1:-1,2:]-2*vi[1:-1, 1:-1]+vi[1:-1,:-2])*self.b/(self.dx**2)
                             +ui[1:-1, 1:-1] - vi[1:-1, 1:-1])

    def Forward(self,n):

        for i in range(n):
            # print("{0}th step done".format(i))
            self.Forward_step()
        
    def BoundaryCondition(self):
        
        self.u[:,0]  = 0.0
        self.u[:,-1] = 0.0
        self.u[0,:]  = 0.0
        self.u[-1,:] = 0.0

        self.v[:,0]  = 0.0
        self.v[:,-1] = 0.0
        self.v[0,:]  = 0.0

    def ReturnValue(self):
        return self.u

    def plot(self,data):
        plt.cla()             
        self.Forward(100)
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        plt.imshow(self.u)

        heatmap = ax.pcolor(self.u, cmap=plt.cm.coolwarm)
                
 #       ax.xaxis.tick_top()
        
        # ax.set_xticklabels(row_labels, minor=False)
        # ax.set_yticklabels(column_labels, minor=False)
        #plt.show()
        
        #ax.plot_wireframe(x,y,self.u, color='r')
        
    def Animation(self):
        
        #ani = animation.FuncAnimation(fig, self.plot, frames=20, interval=50, blit=False)
        ani = animation.FuncAnimation(fig, self.plot, frames=20, interval=100, blit=False)

        ani.save("output.gif", writer="imagemagick")

    
    def Display(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        ax.plot_wireframe(x,y,self.u, color='r')
        ax.set_xlabel('z')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        plt.show()







        
    
if __name__ == "__main__":

    # HD = HeatDiffusion()
    HD = WaveEquation()
    HD.Initialize()
    #HD.Display()

    # for i in range(10):
    #     HD.Forward(10)
    #     HD.Display()

    HD.Animation()

    # RD = ReactionDiffusion()
    # RD.Initialize()
    # # RD.Display()

    # for i in range(10):
    #     RD.Forward(10)
    #     RD.Display()

    # RD.Animation()

    
