import numpy as np
import os 
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d

from ratSimulator import RatSimulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def showGridCells(agent, dataGenerator, num_traj, num_steps, pcu, hcu, llu, bins, place_cell_centers, head_cell_centers):
    factor=2.2/bins
    activityMap=np.zeros((llu, bins, bins))
    counter = 0
    new_activities = np.zeros((llu))


    X=np.zeros((num_traj,num_steps,3))
    positions=np.zeros((num_traj,num_steps,2))
    angles=np.zeros((num_traj,num_steps,1))

    env=RatSimulator(num_steps)

    print(">>Generating trajectory")
    for i in range(num_traj):
        vel, angVel, pos, angle =env.constantTrajectory()
        X[i,:,0]=vel
        X[i,:,1]=np.sin(angVel)
        X[i,:,2]=np.cos(angVel)
        positions[i,:]=pos

    init_X=np.zeros((num_traj,8,pcu + hcu))
    for i in range(8):
        init_X[:, i, :pcu]=dataGenerator.computePlaceCellsDistrib(positions[:,(i*100)], place_cell_centers)
        init_X[:, i, pcu:]=dataGenerator.computeHeadCellsDistrib(angles[:,(i*100)], head_cell_centers)


    print(">>Computing Actvity maps")
    #Feed 500 examples at time to avoid memory problems. Otherwise (10000*100=1million matrix)
    batch_size=500
    for startB in range(0, num_traj, batch_size):
        endB=startB+batch_size

        #Divide the sequence in 100 steps.
        for startT in range(0, num_steps, 100):
            endT=startT+100

            #Retrieve the inputs for the timestep
            xBatch=X[startB:endB, startT:endT]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ agent.X: xBatch, 
                        agent.placeCellGround: init_X[startB:endB, (startT//100), : pcu], 
                        agent.headCellGround: init_X[startB:endB,  (startT//100), pcu :]}
            
            linearNeurons=agent.sess.run(agent.linearLayer, feed_dict=feed_dict)
            #print(linearNeurons.shape)


            #Convert 500,100,2 -> 50000,2
            posReshaped=np.reshape(positions[startB:endB,startT:endT],(-1,2))

            #save the value of the neurons in the linear layer at each timestep
            for t in range(linearNeurons.shape[0]):
                #Now there are the 512 values of the same location
                new_activities+=np.abs(linearNeurons[t])#linearNeurons must be a vector of 512
                #counterActivityMap[:, bin_y, bin_x]+=np.ones((512))
                counter += 1

    #counterActivityMap[counterActivityMap==0]=1
    #Compute average value
    #result=activityMap/counterActivityMap
    result = new_activities / counter
    #print(result.shape)

    os.makedirs("activityMaps", exist_ok=True)
    os.makedirs("corrMaps", exist_ok=True)

    #normalize total or single?
    normMap=(result -np.min(result))/(np.max(result)-np.min(result))
    #adding absolute value
    reshapedMap = np.reshape(normMap, (32,16))
    fig=plt.figure(figsize=(16, 8))
    plt.imshow(reshapedMap, cmap="jet", origin="lower")
    plt.axis('off')

    fig.savefig('activityMaps/neurons.jpg')





