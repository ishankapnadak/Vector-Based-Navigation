import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class Trainer():
    def __init__(self, agent, numSteps):
        self.agent=agent
        #self.PlaceCells_units=pcu
        self.numberSteps=numSteps

    def training(self, X, Y, epoch):

        #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
        feed_dict={ self.agent.X: X, 
                    self.agent.LabelSine: Y[:, :, 0],
                    self.agent.LabelCosine: Y[:, :, 1],
                    #self.agent.placeCellGround: init_LSTM[:, :self.PlaceCells_units], 
                    #self.agent.headCellGround: init_LSTM[:, self.PlaceCells_units:],
                    self.agent.keepProb: 0.5}
        
        _, meanLoss, sineLoss, cosineLoss =self.agent.sess.run([self.agent.opt, self.agent.meanLoss, self.agent.errorSine, self.agent.errorCosine], feed_dict=feed_dict)

        #training epoch finished, save the errors for tensorboard
        mergedData = self.agent.sess.run(self.agent.mergeEpisodeData, feed_dict={self.agent.mn_loss: meanLoss})
        self.agent.file.add_summary(mergedData, epoch)
        
        ##mergedData = self.agent.sess.run(self.agent.mergeNormData, feed_dict={self.agent.norm_loss: normLoss})
        self.agent.file.add_summary(mergedData, epoch)

        mergedData = self.agent.sess.run(self.agent.mergeSineData, feed_dict={self.agent.sine_loss: sineLoss})
        self.agent.file.add_summary(mergedData, epoch)

        mergedData = self.agent.sess.run(self.agent.mergeCosineData, feed_dict={self.agent.cosine_loss: cosineLoss})
        self.agent.file.add_summary(mergedData, epoch)
        

    def testing(self, X, positions_array, epoch):
        #avgDistance = 0

        #displayPredTrajectories=np.zeros((10,800,2))
        home_location = positions_array[:,0]

        #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
        feed_dict={ self.agent.X: X, 
                    #self.agent.placeCellGround: init_X[:, (startB//100), :self.PlaceCells_units], 
                    #self.agent.headCellGround: init_X[:, (startB//100), self.PlaceCells_units:],
                    self.agent.keepProb: 1}
        
        sine, cosine = self.agent.sess.run([self.agent.OutputSine, self.agent.OutputCosine], feed_dict=feed_dict)
        #print(norms.shape)
        
        #retrieve the position in these 100 timesteps
        #print(positions.shape)
        #print(positions)
        home_location = home_location.reshape(10,1,2)
        X_steps = positions_array[:,:,0] - home_location[:,:,0]
        Y_steps = positions_array[:,:,1] - home_location[:,:,1]
        phase_truth = np.pi + np.arctan2(Y_steps, X_steps)
        sine_truth = np.sin(phase_truth).reshape(10*self.numberSteps,1)
        cosine_truth = np.cos(phase_truth).reshape(10*self.numberSteps,1)
        #Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
        #idx=np.argmax(placeCellLayer, axis=1)
        
        #Retrieve the place cell center of the activated place cell
        #predPositions=pcc[idx]

        #Update the predictedTrajectory.png
        #if epoch%8000==0:
            #displayPredTrajectories[:,startB:endB]=np.reshape(predPositions,(10,100,2))

        #Compute the distance between truth position and place cell center
        sine_error = np.mean(np.abs(sine - sine_truth))
        cosine_error = np.mean(np.abs(cosine - cosine_truth))
        
        #testing epoch finished, save the accuracy for tensorboard
        mergedData=self.agent.sess.run(self.agent.mergeSineErrorData, feed_dict={self.agent.sine_error: sine_error})
        
        self.agent.file.add_summary(mergedData, epoch)

        mergedData=self.agent.sess.run(self.agent.mergeCosineErrorData, feed_dict={self.agent.cosine_error: cosine_error})
        
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