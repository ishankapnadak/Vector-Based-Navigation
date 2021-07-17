import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class Trainer():
    def __init__(self, agent, numSteps):
        self.agent=agent
        #self.PlaceCells_units=pcu
        self.numberSteps=numSteps

    def training(self, X, Y, epoch):
        #X = tf.cast(X, tf.float32)
        #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
        #print(Y.dtype)
        feed_dict={self.agent.X: X, self.agent.LabelX: Y[:, :, 0], self.agent.LabelY: Y[:, :, 1], self.agent.keepProb: 0.5}
        
        _, meanLoss, X_loss, Y_loss =self.agent.sess.run([self.agent.opt, self.agent.meanLoss, self.agent.errorX, self.agent.errorY], feed_dict=feed_dict)

        #training epoch finished, save the errors for tensorboard
        mergedData = self.agent.sess.run(self.agent.mergeEpisodeData, feed_dict={self.agent.mn_loss: meanLoss})
        self.agent.file.add_summary(mergedData, epoch)
        
        #mergedData = self.agent.sess.run(self.agent.mergeNormData, feed_dict={self.agent.norm_loss: norm_loss})
        #self.agent.file.add_summary(mergedData, epoch)

        mergedData = self.agent.sess.run(self.agent.mergeXData, feed_dict={self.agent.X_loss: X_loss})
        self.agent.file.add_summary(mergedData, epoch)

        mergedData = self.agent.sess.run(self.agent.mergeYData, feed_dict={self.agent.Y_loss: Y_loss})
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
        
        X,Y = self.agent.sess.run([self.agent.OutputX, self.agent.OutputY], feed_dict=feed_dict)
        #print(norms.shape)
        
        #retrieve the position in these 100 timesteps
        #print(positions.shape)
        #print(positions)
        positions_relative = positions_array - home_location.reshape(10,1,2)
        X_relative = positions_relative[:,:,0]
        Y_relative = positions_relative[:,:,1]
        X_reshaped = X_relative.reshape(10 * self.numberSteps,1)
        Y_reshaped = Y_relative.reshape(10 * self.numberSteps,1)
        #Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
        #idx=np.argmax(placeCellLayer, axis=1)
        
        #Retrieve the place cell center of the activated place cell
        #predPositions=pcc[idx]

        #Update the predictedTrajectory.png
        #if epoch%8000==0:
            #displayPredTrajectories[:,startB:endB]=np.reshape(predPositions,(10,100,2))

        #Compute the distance between truth position and place cell center
        errorX = np.abs(X_reshaped - X)
        avgerrorX = np.mean(errorX)

        errorY = np.abs(Y_reshaped - Y)
        avgerrorY = np.mean(errorY)
        
        #testing epoch finished, save the accuracy for tensorboard
        mergedData=self.agent.sess.run(self.agent.mergeAccuracyXData, feed_dict={self.agent.avgX: avgerrorX})
        self.agent.file.add_summary(mergedData, epoch)

        mergedData=self.agent.sess.run(self.agent.mergeAccuracyYData, feed_dict={self.agent.avgY: avgerrorY})
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