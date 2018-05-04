import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class HeatDiffusion():

    def __init__(self):
        self.Lx = 101
        self.Ly = 101
        self.dt = 1e-5
        self.dx = 1e-2
        self.dy = 1e-2
        self.u = np.zeros((self.Lx,self.Ly),dtype=float)
        self.D = 1.0
        self.time = 0.0
        self.yBoundaryValue = 1.0
        
    def Initialize(self):

        sigma = (self.dx*self.Lx)*0.05
        Gauss = lambda x: np.exp(-(x**2.0/(2.0*sigma**2)))/(np.sqrt(2.0*np.pi*sigma**2))
                
        x, y = np.meshgrid((np.arange(self.Lx)-self.Lx/2)*self.dx,
                           (np.arange(self.Ly)-self.Ly/2)*self.dy)
        self.u = Gauss(x)*Gauss(y)/(Gauss(0)**2)
        
        # for i in range(self.Lx):
        #     for j in range(self.Ly):
        #         self.u[i,j] \
        #             = 1.0 if ((np.abs(i-self.Lx/2)<20) and (np.abs(j-self.Ly/2)<20)) else 0.0
        
                        
    def Forward_step(self):

        ui = self.u.copy()

        self.BoundaryCondition()
        
        self.u[1:-1, 1:-1] \
            += self.dt*self.D*((ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/(self.dx**2)
                               + (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/(self.dy**2))

        self.time += self.dt

    def Forward(self,n):

        for i in range(n):
            # print("{0}th step done".format(i))
            self.Forward_step()
        
    def BoundaryCondition(self):
        
        self.u[0,:]  = 0.0
        self.u[-1,:] = 0.0
        self.u[:,0]  = 0.0
        self.u[:,-1] = self.yBoundaryValue        
        
    def ReturnValue(self):
        return self.u

    def AnalyticalResult(self):

        ### Stationary solution for yBoundaryValue = 1.0 otherwise 0 ###

        sol = np.zeros((self.Lx,self.Ly),dtype=float)
        y, x = np.meshgrid((np.arange(self.Lx))*self.dx,(np.arange(self.Ly))*self.dy)
        for m in range(1,200):
            sol += np.sin((2.0*m-1)*np.pi*x)/(2.0*m-1)*self.yBoundaryValue*(np.sinh(m*np.pi*y)/np.sinh(m*np.pi))
        sol *= 4.0/np.pi

        return sol

    
    
    def Output(self,Mode):

        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        plt.subplots_adjust(left=0.0, bottom=0.05, right=0.95, top=1.0)    

        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))

        NFrames = 60

        def plot(i):
            plt.cla()                      
            self.Forward(100)
            ax.set_zlim(-0.5,1.0)
            ax.text2D(0.05, 0.90,
                      r"$\frac{\partial}{\partial t}u = \left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2} \right)u$",
                  transform=ax.transAxes,size=20)
            ax.text2D(0.05, 0.78,
                      r"$t=$"+str("{:.2f}".format(round(self.time,2))),
                      transform=ax.transAxes,size=20)

            ax.set_xlabel(r'$x$',size=20)
            ax.set_ylabel(r'$y$',size=20)
            ax.set_zlabel(r'$u$',size=20)

            ax.plot_wireframe(x,y,self.u, color='r',rstride=5,cstride=5)
            print("{0} / {1} calculated".format(i+1,NFrames))

        if Mode == "Display":

            for i in range(NFrames):
                plot(i)
                plt.draw()
                plt.pause(0.001)                


            CompareFlag = False
            #CompareFlag = True
            if CompareFlag == True:
                sol = self.AnalyticalResult()
                print("u = ")
                print(self.u)
                print("Stationary Solution = ")
                print(sol)
                print("Diff = ")
                print(self.u-sol)
                plt.cla()                      
                ax.plot_wireframe(x,y,self.u, color='r',rstride=5,cstride=5,label="Numerical")
                ax.plot_wireframe(x,y,sol, color='b',rstride=5,cstride=5,label="Analytical")
                plt.legend()
                plt.savefig('Comparison.png')
                plt.pause(100)

            
        elif Mode == "MakeGif":
            ani = animation.FuncAnimation(fig, plot, frames=NFrames, interval=50, blit=False)
            ani.save("output.gif", writer="imagemagick")        
