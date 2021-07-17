import numpy as np
import os 
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d

from ratSimulator import RatSimulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def showGridCells(agent, dataGenerator, num_traj, num_steps, llu, slu, bins):
    factor=2.2/bins
    activityMap1 = np.zeros((llu, bins, bins))
    activityMap2 = np.zeros((slu, bins, bins))
    counter = 0

    X=np.zeros((num_traj,num_steps,3))
    positions=np.zeros((num_traj,num_steps,2))
    angles=np.zeros((num_traj,num_steps,1))

    env=RatSimulator(num_steps)

    print(">>Generating trajectory")
    for i in range(num_traj):
        vel, angVel, pos, angle =env.generateTrajectory()
        X[i,:,0]=vel
        X[i,:,1]=np.sin(angVel)
        X[i,:,2]=np.cos(angVel)
        positions[i,:]=pos

    #init_X=np.zeros((num_traj,8,pcu + hcu))
    home_locations = positions[:,0]
    #for i in range(8):
        #init_X[:, i, :pcu]=dataGenerator.computePlaceCellsDistrib(positions[:,(i*100)], place_cell_centers)
        #init_X[:, i, pcu:]=dataGenerator.computeHeadCellsDistrib(angles[:,(i*100)], head_cell_centers)


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
                        #agent.placeCellGround: init_X[startB:endB, (startT//100), : pcu], 
                        #agent.headCellGround: init_X[startB:endB,  (startT//100), pcu :]
                        agent.keepProb: 1
                        }
            
            linearNeurons1, linearNeurons2 = agent.sess.run([agent.linearLayer, agent.linearLayer2], feed_dict=feed_dict)
            #print(linearNeurons.shape)
            #-----------------------------------------------------#


            #Convert 500,100,2 -> 50000,2
            posReshaped=np.reshape(positions[startB:endB,startT:endT],(-1,2))

            #save the value of the neurons in the linear layer at each timestep
            for t in range(linearNeurons1.shape[0]):
                #Compute which bins are for each position
                bin_x, bin_y=(posReshaped[t]//factor).astype(int)

                if(bin_y==bins):
                    bin_y=bins-1
                elif(bin_x==bins):
                    bin_x=bins-1

                #Now there are the 512 values of the same location
                activityMap1[:,bin_x, bin_y]+=np.abs(linearNeurons1[t])#linearNeurons must be a vector of 512
                activityMap2[:,bin_x, bin_y]+=np.abs(linearNeurons2[t])
                counter += 1

    #counterActivityMap[counterActivityMap==0]=1
    #Compute average value
    result1 = activityMap1/counter
    result2 = activityMap2/counter

    os.makedirs("activityMaps", exist_ok=True)
    os.makedirs("corrMaps", exist_ok=True)

    #normalize total or single?
    normMap1=(result1 -np.min(result1))/(np.max(result1)-np.min(result1))
    normMap2=(result2 -np.min(result2))/(np.max(result2)-np.min(result2))
    #adding absolute value

    cols=16
    rows=32
    #Save images
    fig=plt.figure(figsize=(80, 80))
    for i in range(llu):
        fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        plt.imshow(normMap1[i], cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('activityMaps/neurons1.jpg')

    #fig=plt.figure(figsize=(80, 80))
    #for i in range(llu):
        #fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        #plt.imshow(correlate2d(normMap[i], normMap[i]), cmap="jet", origin="lower")
        #plt.axis('off')

    #fig.savefig('corrMaps/neurons.jpg')

    cols=8
    rows=16
    #Save images
    fig=plt.figure(figsize=(80, 80))
    for i in range(slu):
        fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        plt.imshow(normMap2[i], cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('activityMaps/neurons2.jpg')





