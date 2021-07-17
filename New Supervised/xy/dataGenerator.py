import numpy as np
import matplotlib.pyplot as plt
import pickle
from ratSimulator import RatSimulator

class dataGenerator():
    def __init__(self, number_steps, num_features):
        #HYPERPARAMETERS
        self.number_steps=number_steps #number of time steps
        self.num_features=num_features #number of features (3 in our case - velocity, sine and cosine of angular velocity)
        #self.placeCell_units=pc_units #number of place cell units
        #self.headCell_units=hd_units #number of head cell units

        self.ratSimulator=RatSimulator(self.number_steps) #initialise rat simulator

        
    def generateData(self, batch_size):
        inputData=np.zeros((batch_size, self.number_steps,self.num_features)) #create list to store input data

        #lists for trajectory data
        velocities=np.zeros((batch_size,self.number_steps)) 
        angVelocities=np.zeros((batch_size,self.number_steps))
        angles=np.zeros((batch_size,self.number_steps))
        positions=np.zeros((batch_size, self.number_steps,2))

        print(">>Generating trajectories")
        for i in range(batch_size): #create as many trajectories as batch size
            vel, angVel, pos, angle=self.ratSimulator.generateTrajectory() #get trajectory data from rat simulator

            #store data of ith trajectory
            velocities[i]=vel
            angVelocities[i]=angVel
            angles[i]=angle
            positions[i]=pos

        #format input data 
        for t in range(self.number_steps):
            inputData[:,t,0] = velocities[:,t]
            inputData[:,t,1] = np.sin(angVelocities[:,t])
            inputData[:,t,2] = np.cos(angVelocities[:,t])

        return inputData, positions, angles #return input data and positions and angles (for supervision)


    def computeX(self, home_location, position):
      X = (position - home_location)[:,0]
      return X

    def computeY(self, home_location, position):
      Y = (position - home_location)[:,0]
      return Y

    def computeAngle(self, home_location, position):
      X = position[:,0] - home_location[:,0]
      Y = position[:,1] - home_location[:,1]
      phase = np.pi + np.arctan2(Y, X)

      return phase