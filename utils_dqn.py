import numpy as np
import pandas as pd
import random
import pickle
import os.path

import tensorflow as tf
import os


class DQN:
    def __init__(self, modelParams, nnDirectory, loadNN, learning_rate = 0.001):
        # Parameters
        self.params = modelParams
        self.learning_rate = learning_rate

        # Network Parameters
        self.num_input = modelParams.stateSize
        self.numActions = modelParams.numActions

        self.inputLayer = tf.placeholder("float", [None, self.num_input]) 
 
        self.outputSingle = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actionSelected = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        if self.params.scopeVarName == '':
            scope = nnDirectory
        else:
            scope = self.params.scopeVarName
        self.numRuns =tf.get_variable(scope + ".numRuns", shape=(), initializer=tf.zeros_initializer())
    
        # Construct network
        if modelParams.type == "DQN_Embedding":
            self.outputLayer = self.build_dqn_withEmbedding(modelParams.nn_Func, scope)
        else:
            self.outputLayer = self.build_dqn(modelParams.nn_Func, scope)

        batch_size = tf.shape(self.inputLayer)[0]

        gather_indices = tf.range(batch_size) * tf.shape(self.outputLayer)[1] + self.actionSelected
        action_predictions = tf.gather(tf.reshape(self.outputLayer, [-1]), gather_indices, name="action_selected_value")
        boundedPrediction = tf.clip_by_value(action_predictions, -1.0, 1.0, name="clipped_action_selected_value")

        # Define loss and optimizer
        lossFunc = tf.squared_difference(self.outputSingle, boundedPrediction)
        self.loss_op = tf.reduce_mean(lossFunc + self.RegularizationFactor(), name="loss_func")

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name="train_func")

        # Initializing session and variables
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        if modelParams.outputGraph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(init)        
        

        self.saver = tf.train.Saver()
        self.directoryName = "./" + nnDirectory
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta):
            if loadNN:
                self.saver.restore(self.sess, self.directoryName)

        self.numStates2Check = len(self.params.states2Monitor)

    # Define the neural network
    def build_dqn(self, NN_Func, scope):
        if NN_Func != None:
            return NN_Func(self.inputLayer, self.numActions, scope)   

        with tf.variable_scope(scope):
            fc1 = tf.contrib.layers.fully_connected(self.inputLayer, 512)
            output = tf.contrib.layers.fully_connected(fc1, self.numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
            
        return output

    # Define the neural network
    def build_dqn_withEmbedding(self, NN_Func, scope):
        
        embedSize = self.params.embeddingInputSize
        restSize = self.params.stateSize - embedSize
        
        embeddingInput = tf.slice(self.inputLayer, [0,0], [-1,embedSize])
        otherInput = tf.slice(self.inputLayer, [0,embedSize], [-1,restSize])
        
        if NN_Func != None:
            return NN_Func(embeddingInput, otherInput, self.numActions, scope)   


        with tf.variable_scope(scope):
            embeddingOut = tf.contrib.layers.fully_connected(embeddingInput, 256)
            hiddenLayerInput = tf.concat([embeddingOut, otherInput], axis = 1)
            hiddenLayerOutput = tf.contrib.layers.fully_connected(hiddenLayerInput, 256)
            output = tf.contrib.layers.fully_connected(hiddenLayerOutput, self.numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
        return output

    def RegularizationFactor(self):
        return 0

    def ExploreProb(self):
        return self.params.ExploreProb(self.numRuns.eval(session = self.sess))
        
    def choose_action(self, observation):
        if np.random.uniform() > self.params.ExploreProb(self.numRuns.eval(session = self.sess)):
            vals = self.outputLayer.eval({self.inputLayer: observation.reshape(1,self.num_input)}, session=self.sess)

            maxArgs = np.argwhere(vals[0] == np.amax(vals[0]))
            a = np.random.choice(maxArgs[0])      
        else:
            a = np.random.randint(0, self.numActions)

        return a

    def actionValuesVec(self, state):
        allVals = self.outputLayer.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess)
        return allVals[0]
        
    def learn(self, s, a, r, s_, terminal):             
        size = len(a)
        
        # calculate (R = r + d * Q(s_))
        rNextState = self.outputLayer.eval({self.inputLayer: s_.reshape(size,self.num_input)}, session=self.sess)
        R = r + np.invert(terminal) * self.params.discountFactor * np.max(rNextState, axis=1)

        idxArray = np.arange(size)
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.random.choice(idxArray, self.params.batchSize)
            feedDict = {self.inputLayer: s[chosen].reshape(self.params.batchSize, self.num_input), self.outputSingle: R[chosen], self.actionSelected: a[chosen]}

            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)

        for i in range(self.numStates2Check):
            state = self.params.states2Monitor[i][0]
            actions2Print = self.params.states2Monitor[i][1]
            print(list(self.actionValuesVec(state)[actions2Print]), end = "\n\n")     

    def save_network(self):
        self.saver.save(self.sess, self.directoryName)
        print("save nn with", self.numRuns.eval(session = self.sess))

    def NumRuns(self):
        return int(self.numRuns.eval(session = self.sess))

    def end_run(self):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)