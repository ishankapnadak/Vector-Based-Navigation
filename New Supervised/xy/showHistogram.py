import numpy as np
import os 
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d

from ratSimulator import RatSimulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def showHistogram(agent, dataGenerator, num_traj, num_steps, llu, slu, tlu, bins):
    #factor=2.2/bins
    #activityMap=np.zeros((llu, bins, bins))
    #counter = 0

    X=np.zeros((num_traj,num_steps,3))
    positions=np.zeros((num_traj,num_steps,2))
    angles=np.zeros((num_traj,num_steps,1))
    hist = np.zeros((400,num_traj))

    env=RatSimulator(num_steps)

    print(">>Generating trajectory")
    for i in range(num_traj):
        vel, angVel, pos, angle =env.generateTrajectory()
        X[i,:,0]=vel
        X[i,:,1]=np.sin(angVel)
        X[i,:,2]=np.cos(angVel)
        positions[i,:]=pos

    #init_X=np.zeros((num_traj,8,pcu + hcu))
    home_location = pos[0]
    #for i in range(8):
        #init_X[:, i, :pcu]=dataGenerator.computePlaceCellsDistrib(positions[:,(i*100)], place_cell_centers)
        #init_X[:, i, pcu:]=dataGenerator.computeHeadCellsDistrib(angles[:,(i*100)], head_cell_centers)


    print(">>Computing Actvity maps")
    #Feed 500 examples at time to avoid memory problems. Otherwise (10000*100=1million matrix)
    batch_size=1
    for startB in range(0, num_traj, batch_size):
        endB=startB+batch_size
        home_X, home_Y = home_location

        #fig=plt.figure(figsize=(12,12))
        #ax=fig.add_subplot(111)
        #ax.set_title("Trajectory agent")
        #ax.plot(pos[:,0], pos[:,1])
        #ax.set_xlim(0,2.2)
        #ax.set_ylim(0,2.2)
        #ax.grid(True)
        #ax.plot(home_X, home_Y, 'o')

        #Divide the sequence in 100 steps.
        
            #print(current_X)


            #Retrieve the inputs for the timestep
        xBatch=X[startB:endB]
        #print(xBatch.shape)

        #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
        feed_dict={ agent.X: xBatch, 
                    #agent.placeCellGround: init_X[startB:endB, (startT//100), : pcu], 
                    #agent.headCellGround: init_X[startB:endB,  (startT//100), pcu :]
                    agent.keepProb : 1
                    }
        
        norm = agent.sess.run([agent.OutputNorm], feed_dict=feed_dict)
        norm = norm[0].reshape((num_steps))
        #print(norm)
        #print(norm.shape)
        #norm_end = norm[99]
        hist[startB] = norm
            
            
    plt.imshow(hist, cmap="inferno", vmin=0, vmax=2.2)
    plt.colorbar()
    plt.savefig('histogram.jpg')
    plt.clf()