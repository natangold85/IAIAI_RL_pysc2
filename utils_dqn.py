import numpy as np
import pandas as pd
import pickle
import os.path

import tensorflow as tf

from utils_tables import ResultFile

class DQN:
    def __init__(self, sizeState, numActions, resultFileName, learning_rate = 0.1, discount_factor = 0.95, numTrials2SaveData = 20, numToWriteResult = 100):
        
        self.history = []
        self.action_history = []
        self.reward_history = []

        # Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_steps = 1000

        # Network Parameters
        self.n_hidden_1 = 256 # 1st layer number of neurons
        self.n_hidden_2 = 256 # 2nd layer number of neurons
        self.num_input = sizeState + 1
        self.numActions = numActions

        self.inputLayer = tf.placeholder("int", [None, self.num_input]) 
        self.outType = tf.placeholder("float", [None, 1])

        # Construct model
        self.build_dqn()

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.outputLayer, labels=self.outType))
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
    def build_dqn(self):
        self.inputLayer = tf.placeholder("int", [None, self.num_input])     
        
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, 1]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(self.inputLayer, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        # Output fully connected layer with a neuron for each class
        self.outputLayer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']


    def choose_action(self, observation):
        return 0

    def learn(self, s, a, r, s_, sToInitValues = None, s_ToInitValues = None):
        s.append(a)
        self.history.append(s)
        self.reward_history.append(r)  
    
    def computeRewards(self, reward):
        for i in range (len(self.reward_history - 2), 0):
            if self.reward_history[i] == 0:
                self.reward_history[i] = self.reward_history[i + 1] * self.discount_factor

    def learnPropogation(self, reward):
        # Define the input function for training
        self.computeRewards(reward)
        self.sess.run([self.train_op, self.loss_op], 
                    feed_dict={self.inputLayer: self.history, self.outputLayer: self.reward_history})
        

    def end_run(self, r):
        saveTable = False
        self.numTrials += 1
        if self.numTrials == self.numTrials2Save:
            saveTable = True
            self.numTrials = 0

        if self.createResultFile:
            self.resultFile.end_run(r, saveTable)

        self.learn(r)

