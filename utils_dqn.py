import numpy as np
import pandas as pd
import pickle
import os.path

import tensorflow as tf

from utils_tables import ResultFile

class NeuralNetwork:
    def __init__(self, featureNames, numActions, nnDirName, resultFileName, learning_rate = 0.1, discount_factor = 0.95, numTrials2SaveData = 20, numToWriteResult = 100):
        
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        #layer_1 = tf.layers.dense(x, n_hidden_1)

        # Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_steps = 1000

        # Network Parameters
        self.n_hidden_1 = 256 # 1st layer number of neurons
        self.n_hidden_2 = 256 # 2nd layer number of neurons
        self.num_input = len(featureNames)
        self.numActions = numActions

        self.inputLayer = tf.placeholder("float", [None, self.num_input])
        self.outputLayer = tf.placeholder("float", [None, self.numActions])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.numActions]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.numActions]))
        }

        # Construct model
        self.model = self.neural_net(self.inputLayer)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.model, labels=self.outputLayer))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.numTrials2Save = numTrials2SaveData
        self.numTrials = 0

        if resultFileName != '':
            self.createResultFile = True
            self.resultFile = ResultFile(resultFileName, numToWriteResult)


    # Define the neural network
    def neural_net(self, x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    def choose_action(self, observation):
        return 0

    def learn(self, s, a, r, s_, sToInitValues = None, s_ToInitValues = None):
        self.state_history.append(s)
        self.action_history.append(a)
        self.reward_history.append(r)  
    
    def computeRewards(self, reward):
        for i in range (len(self.reward_history - 2), 0):
            if self.reward_history[i] == 0:
                self.reward_history[i] = self.reward_history[i + 1] * self.discount_factor

    def backPropagation(self, reward):
        # Define the input function for training
        self.computeRewards(reward)
        self.sess.run([self.train_op, self.loss_op], 
                    feed_dict={self.inputLayer: self.state_history, self.outputLayer: batch_y})
        

    def end_run(self, r):
        saveTable = False
        self.numTrials += 1
        if self.numTrials == self.numTrials2Save:
            saveTable = True
            self.numTrials = 0

        if self.createResultFile:
            self.resultFile.end_run(r, saveTable)

        self.backPropagation(r)

