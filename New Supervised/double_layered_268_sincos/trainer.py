import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class Trainer():
    def __init__(self, agent, numSteps):
        self.agent=agent
        #self.PlaceCells_units=pcu
        self.numberSteps=numSteps

    def training(self, X, Y, epoch):

        #Stores the means of the losses among a training epoch.
        #Used to show the stats on tensorboard
        mn_loss=0

        #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
        for startB in range(0, self.numberSteps, 100):
            endB=startB+100

            #Retrieve the inputs for the 100 timesteps
            xBatch=X[:,startB:endB]
            
            #Retrieve the labels for the 100 timesteps
            yBatch=Y[:,startB:endB]

            #Retrieve label at timestep 0 for the 100 timesteps
            #init_LSTM=yBatch[:,0]
            #print(yBatch.shape)
            #print(yBatch)

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ self.agent.X: xBatch, 
                        self.agent.LabelNorm: yBatch[:, :, 0],
                        self.agent.LabelSine: yBatch[:, :, 1],
                        self.agent.LabelCosine: yBatch[:, :, 2],
                        #self.agent.placeCellGround: init_LSTM[:, :self.PlaceCells_units], 
                        #self.agent.headCellGround: init_LSTM[:, self.PlaceCells_units:],
                        self.agent.keepProb: 0.5}
            
            _, meanLoss, NormLoss, SineLoss, CosineLoss=self.agent.sess.run([self.agent.opt,
                                                                  self.agent.meanLoss,
                                                                  self.agent.errorNorm,
                                                                  self.agent.errorSine,
                                                                  self.agent.errorCosine], feed_dict=feed_dict)

            mn_loss += meanLoss/(self.numberSteps//100)

        #training epoch finished, save the errors for tensorboard
        mergedData = self.agent.sess.run(self.agent.mergeEpisodeData, feed_dict={self.agent.mn_loss: mn_loss})
        
        self.agent.file.add_summary(mergedData, epoch)

    def testing(self, X, positions_array, epoch):
        avgDistance = 0

        #displayPredTrajectories=np.zeros((10,800,2))
        home_location = positions_array[:,0]

        #Divide the sequence in 100 steps
        for startB in range(0, self.numberSteps, 100):
            endB=startB+100

            #Retrieve the inputs for the timestep
            xBatch = X[:,startB:endB]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ self.agent.X: xBatch, 
                        #self.agent.placeCellGround: init_X[:, (startB//100), :self.PlaceCells_units], 
                        #self.agent.headCellGround: init_X[:, (startB//100), self.PlaceCells_units:],
                        self.agent.keepProb: 1}
            
            norms = self.agent.sess.run(self.agent.OutputNorm, feed_dict=feed_dict)
            #print(norms.shape)
            
            #retrieve the position in these 100 timesteps
            positions = positions_array[:,startB:endB]
            #print(positions.shape)
            #print(positions)
            positions_relative = positions - home_location.reshape(10,1,2)
            norms_truth = np.linalg.norm(positions_relative, axis=2)
            norms_truth = norms_truth.reshape(1000,1)
            #Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
            #idx=np.argmax(placeCellLayer, axis=1)
            
            #Retrieve the place cell center of the activated place cell
            #predPositions=pcc[idx]

            #Update the predictedTrajectory.png
            #if epoch%8000==0:
                #displayPredTrajectories[:,startB:endB]=np.reshape(predPositions,(10,100,2))

            #Compute the distance between truth position and place cell center
            distances= np.abs(norms - norms_truth)
            avgDistance +=np.mean(distances)/(self.numberSteps//100)
        
        #testing epoch finished, save the accuracy for tensorboard
        mergedData=self.agent.sess.run(self.agent.mergeAccuracyData, feed_dict={self.agent.avgD: avgDistance})
        
        self.agent.file.add_summary(mergedData, epoch)

        #Compare predicted trajectory with real trajectory
        #if epoch%8000==0:
            #rows=3
            #cols=3
            #fig=plt.figure(figsize=(40, 40))
            #for i in range(rows*cols):
                #ax=fig.add_subplot(rows, cols, i+1)
                #plot real trajectory
                #plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b', label="Truth Path")
                #plot predicted trajectory
                #plt.plot(displayPredTrajectories[i,:,0], displayPredTrajectories[i,:,1], 'go', label="Predicted Path")
                #plt.legend()
                #ax.set_xlim(0,2.2)
                #ax.set_ylim(0,2.2)

            #fig.savefig('predictedTrajectory.png')