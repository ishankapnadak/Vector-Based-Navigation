import numpy as np 
import matplotlib.pyplot as plt
class RatSimulator():
    def __init__(self, n_steps):
        self.number_steps=n_steps #initialise number of time steps
        self.dt=0.02 #resolution of time steps
        self.maxGap=2.17 #maximum position before detecting a wall
        self.minGap=0.03 #minimum position before detecting a wall

        self.velScale=0.13 #scale of Rayleigh distribution for forward velocity
        self.mAngVel=0 #mean angular velocity
        self.stddevAngVel=330 #standard deviation of angular velocity


    #function to generate a random trajectory
    def generateTrajectory(self):
        velocities=np.zeros((self.number_steps)) #initialise velocities list
        angVel=np.zeros((self.number_steps)) #initialise angular velocities list
        positions=np.zeros((self.number_steps, 2)) #initialise positions list
        angle=np.zeros((self.number_steps)) #initialise facing angles list

        for t in range(self.number_steps): #iterate
            #Initialize the agent randomly in the environment
            if(t==0):
                pos=np.random.uniform(low=0, high=2.2, size=(2)) #sample position from uniform distribution
                facAng=np.random.uniform(low=-np.pi, high=np.pi) #sample facing angle from uniform distribution
                prevVel=0 #set previous velocity = 0

            #Check if the agent is near a wall
            if(self.checkWallAngle(facAng, pos)): #checkWallAngle is a helper function
                #if True, calculate the direction in which to turn by 90 degrees
                rotVel=np.deg2rad(np.random.normal(self.mAngVel, self.stddevAngVel))
                dAngle=self.computeRot(facAng, pos) + rotVel*0.02 #compute change in angle
                #Velocity reduction factor = 0.25
                vel=np.squeeze(prevVel - (prevVel*0.25))
            #If the agent is not near a wall, randomly sample velocity and angVelocity
            else:
                #Sampling velocity
                vel=np.random.rayleigh(self.velScale) #Rayleigh
                #Sampling angular velocity
                rotVel=np.deg2rad(np.random.normal(self.mAngVel, self.stddevAngVel)) #Gaussian
                dAngle=rotVel*0.02 #change in angle

            #Update the position of the agent
            newPos=pos + (np.asarray([np.cos(facAng), np.sin(facAng)])*vel)*self.dt
            
            #Update the facing angle of the agent
            newFacAng=(facAng + dAngle)
            #Keep the orientation between -np.pi and np.pi
            if(np.abs(newFacAng)>=(np.pi)):     
                newFacAng=-1*np.sign(newFacAng)*(np.pi - (np.abs(newFacAng)- np.pi))

            #store quantities in respective lists
            velocities[t]=vel
            angVel[t]=rotVel
            positions[t]=pos
            angle[t]=facAng
            
            pos=newPos
            facAng=newFacAng
            prevVel=vel
        
        '''
        #USED TO DISPLAY THE TRAJECTORY ONCE FINISHED
        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(111)
        ax.set_title("Trajectory agent")
        ax.plot(positions[:,0], positions[:,1])
        ax.set_xlim(0,2.2)
        ax.set_ylim(0,2.2)
        ax.grid(True)

        plt.show()
        '''
        
        return velocities, angVel, positions, angle #return all four lists of the trajectory

    #HELPER FUNCTIONS
    def checkWallAngle(self, ratAng, pos):
        #This checks the position of the rat according to the quadrant it's facing towards
        if((0<=ratAng and ratAng<=(np.pi/2)) and np.any(pos>self.maxGap)):
          return True
        elif((ratAng>=(np.pi/2) and ratAng<=np.pi) and (pos[0]<self.minGap or pos[1]>self.maxGap)):
          return True
        elif((ratAng>=-np.pi and ratAng<=(-np.pi/2)) and np.any(pos<self.minGap)):
          return True
        elif((ratAng>=(-np.pi/2) and ratAng<=0) and (pos[0]>self.maxGap or pos[1]<self.minGap)):
          return True
        else:
          return False
    
    def computeRot(self,ang, pos):
        rot=0
        if(ang>=0 and ang<=(np.pi/2)):
          if(pos[1]>self.maxGap):
            rot=-ang
          elif(pos[0]>self.maxGap):
            rot=np.pi/2-ang
        elif(ang>=(np.pi/2) and ang<=np.pi):
          if(pos[1]>self.maxGap):
            rot=np.pi-ang
          elif(pos[0]<self.minGap):
            rot=np.pi/2 -ang
        elif(ang>=-np.pi and ang<=(-np.pi/2)):
          if(pos[1]<self.minGap):
            rot=-np.pi - ang
          elif(pos[0]<self.minGap):
            rot=-(ang + np.pi/2)
        else:
          if(pos[1]<self.minGap):
            rot=-ang
          elif(pos[0]>self.maxGap):
            rot=(-np.pi/2) - ang

        return rot