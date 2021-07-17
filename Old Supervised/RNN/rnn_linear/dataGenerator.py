import numpy as np
import matplotlib.pyplot as plt
import pickle
from ratSimulator import RatSimulator

class dataGenerator():
    def __init__(self, number_steps, num_features, pc_units, hd_units):
        #HYPERPARAMETERS
        self.number_steps=number_steps #number of time steps
        self.num_features=num_features #number of features (3 in our case - velocity, sine and cosine of angular velocity)
        self.placeCell_units=pc_units #number of place cell units
        self.headCell_units=hd_units #number of head cell units

        self.ratSimulator=RatSimulator(self.number_steps) #initialise rat simulator

        
    def generateData(self, batch_size):
        inputData=np.zeros((batch_size, self.number_steps,3)) #create list to store input data

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
        

    def computePlaceCellsDistrib(self, positions, cellCenters):
        num_cells=cellCenters.shape[0] #number of place cells
        batch_size=positions.shape[0] #batch size or number of trajectories
        #Place Cell scale
        sigma=0.01

        summs=np.zeros(batch_size)
        #Every row stores the distribution for a trajectory
        distributions=np.zeros((batch_size,num_cells))
        #We have 256 elements in the Place Cell Distribution. For each of them
        for i in range(num_cells):
            #compute the sum of all Gaussians
            l2Norms=np.sum((positions - cellCenters[i])**2, axis=1)
            placeCells=np.exp(-(l2Norms/(2*sigma**2)))

            distributions[:,i]=placeCells #store ith Gaussian in distribution corresponding to ith place cell
            summs +=placeCells

        distributions=distributions/summs[:,None] #normalise the distribution of place cells
        #This returns a (batch_size x num_cells) matrix where (i,j) corresponds to probability of jth place cell in ith trajectory
        return distributions 

    def computeHeadCellsDistrib(self,facingAngles, cellCenters):
        num_cells=cellCenters.shape[0] #number of head cells
        batch_size=facingAngles.shape[0] #batch size or number of trajectories
        #Concentration parameter 
        k=20

        summs=np.zeros(batch_size)
        #Every row stores the distribution for a trajectory
        distributions=np.zeros((batch_size,num_cells))
        #We have 12 elements in the Head Direction Cell Distribution. For each of them
        for i in range(num_cells):
            #compute the distribution of head cells
            headDirects=np.squeeze(np.exp(k*np.cos(facingAngles - cellCenters[i])))
            distributions[:,i]=headDirects #store ith term in distribution corresponding to ith head cell
            summs+=headDirects
        
        distributions=distributions/summs[:,None] #normalise the distribution of head cells
        #This returns a (batch_size x num_cells) matrix where (i,j) corresponds to probability of jth place cell in ith trajectory
        return distributions