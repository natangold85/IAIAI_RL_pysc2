import numpy as np
import pandas as pd
import random
import pickle
import os.path

import tensorflow as tf
import os

class DQN:
    def __init__(self, numActions, sizeState, nnDirectory, terminalStates, NN_Func = None, discount_factor = 0.95, explorationProb = 0.9, learning_rate = 0.1):
        # Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.explorationProb = explorationProb

        # Network Parameters
        self.terminalStates = terminalStates
        self.num_input = sizeState
        self.numActions = numActions

        self.inputLayer = tf.placeholder("float", [None, self.num_input]) 
        
        self.outputSingle = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actionSelected = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer())

        # Construct network
        self.outputLayer = self.build_dqn(NN_Func)

        batch_size = tf.shape(self.inputLayer)[0]

        gather_indices = tf.range(batch_size) * tf.shape(self.outputLayer)[1] + self.actionSelected
        action_predictions = tf.gather(tf.reshape(self.outputLayer, [-1]), gather_indices)
        boundedPrediction = tf.minimum(tf.maximum(action_predictions, -1.0), 1.0)

        # Define loss and optimizer
        lossFunc = tf.squared_difference(self.outputSingle, boundedPrediction)
        self.loss_op = tf.reduce_mean(lossFunc + self.RegularizationFactor())

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # Initializing session and variables
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)        
        

        self.directoryName = nnDirectory
        self.saver = tf.train.Saver()
        fnameNNMeta = nnDirectory + ".meta"
        if os.path.isfile(fnameNNMeta):
            tf.reset_default_graph()
            self.saver.restore(self.sess, nnDirectory)


    # Define the neural network
    def build_dqn(self, NN_Func):
        if NN_Func != None:
            return NN_Func(self.inputLayer)   

        # Store layers weight & bias
        n_hidden_1 = 512
        n_hidden_2 = 256

        # self.weights = {
        #     'h1': tf.Variable(tf.random_normal([self.num_input, n_hidden_1])),
        #     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        #     'out': tf.Variable(tf.random_normal([n_hidden_2, self.numActions]))
        # }
        # self.biases = {
        #     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #     'out': tf.Variable(tf.random_normal([self.numActions]))
        # }
        layer_1 = tf.contrib.layers.fully_connected(self.inputLayer, n_hidden_1)
        output = tf.contrib.layers.fully_connected(layer_1, self.numActions)
        # Hidden fully connected layer with 256 neurons
        # layer_1 = tf.add(tf.matmul(self.inputLayer, self.weights['h1']), self.biases['b1'])
        # # Hidden fully connected layer with 256 neurons
        # layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class range between (-1,1)
        # output = tf.nn.sigmoid(tf.matmul(layer_2, self.weights['out'] + self.biases['out'])) * 2 - 1
        return output

    def RegularizationFactor(self):
        return 0

    def choose_action(self, observation):
        if np.random.uniform() > self.explorationProb:
            vals = self.outputLayer.eval({self.inputLayer: observation.reshape(1,self.num_input)}, session=self.sess)
            a = np.argmax(vals[0])      
        else:
            a = random.randint(0, self.numActions - 1)

        return a

    def learn(self, s, a, r, s_):     
        size = len(a)
        rNextState = self.outputLayer.eval({self.inputLayer: s_.reshape(size, self.num_input)}, session=self.sess)
        
        for i in range(size):
            if not self.TerminalState(s_[i,:]):
                r[i] += self.discount_factor * np.max(rNextState[i,:])
                
        feedDict = {self.inputLayer: s.reshape(size, self.num_input), self.outputSingle: r, self.actionSelected: a}
        self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)

    def save_network(self):
        self.saver.save(self.sess, self.directoryName)
        print("save nn with", self.numRuns.eval(session = self.sess))

    def end_run(self):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)

    def TerminalState(self, s):
        for v in self.terminalStates.values():
            if np.array_equal(s, v):
                return True
        return False

