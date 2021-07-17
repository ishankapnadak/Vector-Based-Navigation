import numpy as np
import os 
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d
import pandas as pd
from sklearn.metrics import r2_score

from ratSimulator import RatSimulator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def showVectors(agent, dataGenerator, num_traj, num_steps, llu, slu, tlu, bins):
    #factor=2.2/bins
    #activityMap=np.zeros((llu, bins, bins))
    #counter = 0

    data = pd.DataFrame(columns=["Norm True", "Norm Predicted"])

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

        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(111)
        ax.set_title("Trajectory agent")
        ax.plot(pos[:,0], pos[:,1])
        ax.set_xlim(0,2.2)
        ax.set_ylim(0,2.2)
        ax.grid(True)
        ax.plot(home_X, home_Y, 'o')

        xBatch=X[startB:endB]

        #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
        feed_dict={ agent.X: xBatch, 
                    #agent.placeCellGround: init_X[startB:endB, (startT//100), : pcu], 
                    #agent.headCellGround: init_X[startB:endB,  (startT//100), pcu :]
                    agent.keepProb : 1
                    }
        
        norms = agent.sess.run([agent.OutputNorm], feed_dict=feed_dict)
        norms = norms[0].reshape((num_steps))

        for i in range(0, num_steps):
        	current_X, current_Y = pos[i]
        	index = i
        	norm = norms[i]
        	norm_true = round(np.linalg.norm((home_location - pos[i])), 3)
        	norm_pred = round(norm, 3)
        	#ax.plot(current_X, current_Y, 'o')
        	phase_true = np.pi + np.arctan2((current_Y - home_Y), (current_X - home_X))
        	data.at[index, "Norm True"] = norm_true
        	data.at[index, "Norm Predicted"] = norm_pred
        	#X_direct = norm * np.cos(phase_true)
        	#Y_direct = norm * np.sin(phase_true)
        	#ax.quiver(current_X, current_Y, X_direct, Y_direct, width=0.005, angles='xy', scale_units='xy', scale=1, color='g')

    fig.savefig('trajectory.jpg')
    data.to_csv('trajectory_data.csv')

    norm_true = data["Norm True"].to_list()
    norm_pred = data["Norm Predicted"].to_list()

    r2 = r2_score(norm_true, norm_pred)
    r2_str = "R2 Score for Norm is: " + str(r2)
    file = open("trajectory_metrics.txt", "w")
    file.write(r2_str)
    file.close()