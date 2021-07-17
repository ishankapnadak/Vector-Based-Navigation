import tensorflow as tf 

#Define the agent structure network
class Network():
    def __init__(self, session, lr, hu, slu, lu, clipping, weightDecay, batch_size, num_features, n_steps):
        self.sess=session
        self.epoch=tf.Variable(0, trainable=False)
        #HYPERPARAMETERS
        self.learning_rate=lr #learning rate
        self.Hidden_units=hu #number of hidden units
        self.LinearLayer_units=lu #number of linear layer units
        self.SecondLayer_units = slu
        #self.PlaceCells_units=place_units #number of place cell units
        #self.HeadCells_units=head_units #number of head cell units
        self.clipping=clipping #gradient clipping
        self.weight_decay=tf.constant(weightDecay, dtype=tf.float32) #weight decay
        self.batch_size=batch_size #batch size
        self.num_features=num_features #number of features

        self.buildNetwork()
        self.buildTraining()
        self.buildTensorBoardStats()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver=tf.compat.v1.train.Saver()
        self.file=tf.compat.v1.summary.FileWriter("tensorboard/", self.sess.graph)

    def buildNetwork(self):
        self.X=tf.compat.v1.placeholder(tf.float32, shape=[None, 100, self.num_features], name="input") #placeholder for input

        self.keepProb=tf.compat.v1.placeholder(tf.float32, name="keep_prob") #placeholder for dropout probability

        #self.placeCellGround=tf.compat.v1.placeholder(tf.float32, shape=[None, self.PlaceCells_units], name="Ground_Truth_Place_Cell") #placeholder for ground-truth of place cell
        #self.headCellGround=tf.placeholder(tf.float32, shape=[None, self.HeadCells_units], name="Ground_Truth_Head_Cell") #placeholder for ground-truth of head cell

        #with tf.compat.v1.variable_scope("LSTM_initialization"):
            #Initialize the Hidden state and Cell state of the LSTM unit using feeding the Ground Truth Distribution at timestep 0. Both have size [batch_size, Hidden_units]
            #self.Wcp=tf.compat.v1.get_variable("Initial_state_cp", [self.PlaceCells_units,self.Hidden_units], initializer=tf.contrib.layers.xavier_initializer())
            #self.Wcd=tf.compat.v1.get_variable("Initial_state_cd", [self.HeadCells_units,self.Hidden_units],  initializer=tf.contrib.layers.xavier_initializer())
            #self.Whp=tf.compat.v1.get_variable("Hidden_state_hp",  [self.PlaceCells_units,self.Hidden_units], initializer=tf.contrib.layers.xavier_initializer())
            #self.Whd=tf.compat.v1.get_variable("Hidden_state_hd",  [self.HeadCells_units,self.Hidden_units],  initializer=tf.contrib.layers.xavier_initializer())

            #Compute self.hidden_state 
            #self.hidden_state= tf.matmul(self.placeCellGround, self.Whp) + tf.matmul( self.headCellGround, self.Whd)
            #Compute self.cell_state
            #self.cell_state=tf.matmul(self.placeCellGround, self.Whp) + tf.matmul( self.headCellGround, self.Whd)

            #Store self.cell_state and self.hidden_state tensors as elements of a single list.
            #If is going to be timestep=0, initialize the hidden and cell state using the Ground Truth Distributions. 
            #Otherwise, use the hidden state and cell state from the previous timestep passed using the placeholders  

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! switch the two !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     
            #self.LSTM_state=tf.compat.v1.nn.rnn_cell.LSTMStateTuple(self.hidden_state, self.cell_state) 
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! switch the two !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   

        with tf.compat.v1.variable_scope("RNN"):
            #Define the single LSTM cell with the number of hidden units
            #self.lstm_cell=tf.contrib.rnn.LSTMCell(self.Hidden_units, name="LSTM_Cell")
            self.rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(self.Hidden_units, name="RNN_Cell")

            #Feed an input of shape [batch_size, 100, features]
            #self.output is a tensor of shape [batch_size, 100, hidden_units]
            #out_hidden_statesTuple is a list of 2 elements: self.cell_state, self.hidden_state where self.output[:, -1, :]=self.cell_state
            self.output, self.hiddenstatevalue = tf.compat.v1.nn.dynamic_rnn(cell=self.rnn_cell, inputs=self.X, dtype=tf.float32)
            # initial_state=self.hidden_state

        with tf.compat.v1.variable_scope("Linear_Decoder"):
            self.W1=tf.compat.v1.get_variable("Weights_LSTM_LinearDecoder", [self.Hidden_units, self.LinearLayer_units], initializer=tf.contrib.layers.xavier_initializer())
            self.B1=tf.compat.v1.get_variable("Biases_LSTM_LinearDecoder", [self.LinearLayer_units], initializer=tf.contrib.layers.xavier_initializer())
            
            #we can't feed a tensor of shape [10,100,128] to the linear layer. We treat each timestep in every trajectory as an example
            #we now have a matrix of shape [100*100,128] which can be fed to the linear layer. The result is the same as 
            #looping 100 times through each timestep examples.
            self.reshapedOut=tf.reshape(self.output, (-1, self.Hidden_units))

            self.linearLayer=tf.matmul(self.reshapedOut, self.W1) + self.B1
            
            #Compute Linear layer and apply dropout
            self.linearLayerDrop=tf.compat.v1.nn.dropout(self.linearLayer, self.keepProb)

        with tf.compat.v1.variable_scope("Linear_Decoder_2"):
            self.W2=tf.compat.v1.get_variable("Weights_Linear_to_Linear", [self.LinearLayer_units, self.SecondLayer_units], initializer=tf.contrib.layers.xavier_initializer())
            self.B2=tf.compat.v1.get_variable("Biases_LSTM_Linear_to_Linear", [self.SecondLayer_units], initializer=tf.contrib.layers.xavier_initializer())
            
            #we can't feed a tensor of shape [10,100,128] to the linear layer. We treat each timestep in every trajectory as an example
            #we now have a matrix of shape [100*100,128] which can be fed to the linear layer. The result is the same as 
            #looping 100 times through each timestep examples.
            #self.reshapedOut=tf.reshape(self.output, (-1, self.Hidden_units))

            self.linearLayer2 = tf.matmul(self.linearLayerDrop, self.W2) + self.B2
            
            #Compute Linear layer and apply dropout
            #self.linearLayer2Drop=tf.compat.v1.nn.dropout(self.linearLayer2, self.keepProb)

        with tf.compat.v1.variable_scope("Norm"):
            self.W3=tf.compat.v1.get_variable("Weights_LinearDecoder_Norm", [self.SecondLayer_units, 1], initializer=tf.contrib.layers.xavier_initializer())
            self.B3=tf.compat.v1.get_variable("Biases_LinearDecoder_Norm", [1], initializer=tf.contrib.layers.xavier_initializer())
            
            #Compute the predicted Place Cells Distribution
            self.OutputNorm=tf.matmul(self.linearLayer2, self.W3) + self.B3

        with tf.compat.v1.variable_scope("Sine"):
            self.W4=tf.compat.v1.get_variable("Weights_LinearDecoder_Sine", [self.SecondLayer_units, 1], initializer=tf.contrib.layers.xavier_initializer())
            self.B4=tf.compat.v1.get_variable("Biases_LinearDecoder_Sine", [1], initializer=tf.contrib.layers.xavier_initializer())   
            
            #Compute the predicted Head-direction Cells Distribution
            self.OutputSine=tf.matmul(self.linearLayer2, self.W4) + self.B4

        with tf.compat.v1.variable_scope("Cosine"):
            self.W5=tf.compat.v1.get_variable("Weights_LinearDecoder_Cosine", [self.SecondLayer_units, 1], initializer=tf.contrib.layers.xavier_initializer())
            self.B5=tf.compat.v1.get_variable("Biases_LinearDecoder_Cosine", [1], initializer=tf.contrib.layers.xavier_initializer())   
            
            #Compute the predicted Head-direction Cells Distribution
            self.OutputCosine=tf.matmul(self.linearLayer2, self.W5) + self.B5
  
    def buildTraining(self):
        #Fed the Ground Truth Place Cells Distribution and Head Direction Cells Distribution
        self.LabelNorm = tf.compat.v1.placeholder(tf.float32, shape=[None, 100], name="Label_Norm")
        self.LabelSine = tf.compat.v1.placeholder(tf.float32,  shape=[None, 100], name="Label_Sine")
        self.LabelCosine = tf.compat.v1.placeholder(tf.float32,  shape=[None, 100], name="Label_Cosine")
        
        self.reshapedNorm = tf.reshape(self.LabelNorm, (-1, 1))
        self.reshapedSine = tf.reshape(self.LabelSine, (-1, 1))
        self.reshapedCosine = tf.reshape(self.LabelCosine, (-1, 1))

        #Compute the errors for each neuron in each trajectory for each timestep [1000,256] and [1000,12] errors
        self.errorNorm = tf.compat.v1.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=self.reshapedNorm, predictions=self.OutputNorm))
        self.errorSine = tf.compat.v1.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=self.reshapedSine, predictions=self.OutputSine))
        self.errorCosine = tf.compat.v1.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=self.reshapedCosine, predictions=self.OutputCosine))
        
        #Convert back the tensor from [1000, 1] to [10,100]
        #self.reshapedErrors=tf.reshape((self.errorPlaceCells + self.errorHeadCells), (-1,100))
        #Compute the truncated backprop error for each trajectory (SUMMING THE ERRORS). [10,100] -> [10,1]
        #self.truncErrors=tf.reduce_sum(self.reshapedErrors, axis=1)

        #Compute the l2_loss
        l2_loss = self.weight_decay*tf.compat.v1.nn.l2_loss(self.W4) + self.weight_decay*tf.compat.v1.nn.l2_loss(self.W3) + self.weight_decay*tf.compat.v1.nn.l2_loss(self.W5)

        #Compute mean among truncated errors [10,1] -> [1] (mean error)
        #self.meanLoss=tf.reduce_mean(self.truncErrors, name="mean_error") + l2_loss
        self.meanLoss = self.errorNorm + self.errorSine + self.errorCosine + l2_loss
        
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate, momentum=0.9)

        self.gvs = self.optimizer.compute_gradients(self.meanLoss)

        #Apply gradient clipping to parameters: Place Cells units (weights, biases) , Head Cells units (weights, biases)
        self.gvs[-4] = [tf.clip_by_value(self.gvs[-4][0], -self.clipping, self.clipping), self.gvs[-4][1]]
        self.gvs[-3] = [tf.clip_by_value(self.gvs[-3][0], -self.clipping, self.clipping), self.gvs[-3][1]]
        self.gvs[-2] = [tf.clip_by_value(self.gvs[-2][0], -self.clipping, self.clipping), self.gvs[-2][1]]
        self.gvs[-1] = [tf.clip_by_value(self.gvs[-1][0], -self.clipping, self.clipping), self.gvs[-1][1]]

        self.opt = self.optimizer.apply_gradients(self.gvs)
    
    def buildTensorBoardStats(self):
        #Episode data
        self.mn_loss = tf.compat.v1.placeholder(tf.float32)
        self.mergeEpisodeData = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar("mean_loss", self.mn_loss)])

        self.avgD = tf.compat.v1.placeholder(tf.float32)
        self.mergeAccuracyData = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar("average_distance", self.avgD)])
    
    def save_restore_Model(self, restore, epoch=0):
        if restore:
            self.saver.restore(self.sess, "agentBackup/graph.ckpt")
        else:
            self.sess.run(self.epoch.assign(epoch))
            self.saver.save(self.sess, "agentBackup/graph.ckpt")