import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



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


    def Output(self,Mode):

        fig = plt.figure()
        ax = fig.add_subplot(111)        
        plt.subplots_adjust(left=0.0, bottom=0.05, right=0.95, top=0.95)

        NFrames = 10
        #NFrames = 20
        x, y = np.meshgrid(np.arange(self.Lx), np.arange(self.Ly))
        def plot(i):
            plt.cla()             
            self.Forward(100)
            plt.imshow(self.u)
            heatmap = ax.pcolor(self.u, cmap=plt.cm.coolwarm)        
            print("{0} / {1} calculated".format(i+1,NFrames))


        
        if Mode == "Display":
            for i in range(NFrames):
                plot(i)
                plt.draw()
                plt.pause(0.001)

        elif Mode == "MakeGif":
            ani = animation.FuncAnimation(fig, plot, frames=NFrames, interval=100, blit=False)
            ani.save("output.gif", writer="imagemagick")       

        

