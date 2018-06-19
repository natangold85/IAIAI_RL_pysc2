import numpy as np
import pandas as pd
import random
import pickle
import os.path

import tensorflow as tf
import os
import sys

class DQN:
    def __init__(self, modelParams, nnDirectory, learning_rate = 0.001):
        # Parameters
        self.params = modelParams
        self.learning_rate = learning_rate

        # Network Parameters
        self.num_input = modelParams.stateSize
        self.numActions = modelParams.numActions

        if self.params.isStateProcessed:
            self.inputLayer = tf.placeholder("float", [None, self.num_input]) 
        else:
            self.inputLayer = tf.placeholder("float", [None, self.num_input, self.num_input]) 
        
        self.outputSingle = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actionSelected = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer())

        # Construct network
        self.outputLayer = self.build_dqn(modelParams.nn_Func, nnDirectory)

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
        

        self.directoryName = "./" + nnDirectory
        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta):
            tf.reset_default_graph()
            if "newDQN" not in sys.argv:
                print("\n\nload network\n\n")
                self.saver.restore(self.sess, self.directoryName)
            else:
                print("\n\nnew network\n\n")

        self.numStates2Check = 5
        self.state2Monitor = np.zeros((self.numStates2Check, modelParams.stateSize), dtype=np.int32, order='C')
        idxSelfLoc = [1, 25, 35, 96, 70]
        idxBeaconLoc = [0, 15, 0, 86, 10]
        for i in range(self.numStates2Check):
            self.state2Monitor[i,idxSelfLoc[i]] = 1
            self.state2Monitor[i,idxBeaconLoc[i]] = 1

    # Define the neural network
    def build_dqn(self, NN_Func, scope):
        if NN_Func != None:
            return NN_Func(self.inputLayer, self.numActions, scope)   

        with tf.variable_scope(scope):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(self.inputLayer, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

            # Fully connected layers
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            output = tf.contrib.layers.fully_connected(fc1, self.numActions)
        return output

    def RegularizationFactor(self):
        return 0

    def choose_action(self, observation):
        if np.random.uniform() > self.params.ExplorationProb(self.numRuns.eval(session = self.sess)):
            if self.params.isStateProcessed:
                vals = self.outputLayer.eval({self.inputLayer: observation.reshape(1,self.num_input)}, session=self.sess)
            else:
                vals = self.outputLayer.eval({self.inputLayer: observation.reshape(1,self.num_input,self.num_input)}, session=self.sess)

            maxArgs = np.argwhere(vals[0] == np.amax(vals[0]))
            a = np.random.choice(maxArgs[0])      
        else:
            a = np.random.randint(0, self.numActions)

        return a

    def actionValuesVec(self, state):
        allVals = self.outputLayer.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess)
        return allVals[0]
        
    def learn(self, s, a, r, s_, terminal):  
        prevVal = []
        for i in range(self.numStates2Check):
            state = self.state2Monitor[i,:]
            prevVal.append(list(self.actionValuesVec(state))) 
              
        size = len(a)
        if self.params.isStateProcessed:
            rNextState = self.outputLayer.eval({self.inputLayer: s_.reshape(size,self.num_input)}, session=self.sess)
        else:
            rNextState = self.outputLayer.eval({self.inputLayer: s_.reshape(size,self.num_input,self.num_input)}, session=self.sess)
        
        # calculate (R = r + d * Q(s_))
        R = r + np.invert(terminal) * self.params.discountFactor * np.max(rNextState, axis=1)
        idxArray = np.arange(size)
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.random.choice(idxArray, self.params.batchSize)

            if self.params.isStateProcessed:
                feedDict = {self.inputLayer: s[chosen].reshape(self.params.batchSize, self.num_input), self.outputSingle: R[chosen], self.actionSelected: a[chosen]}
            else:      
                feedDict = {self.inputLayer: s[chosen].reshape(self.params.batchSize, self.num_input, self.num_input), self.outputSingle: R[chosen], self.actionSelected: a[chosen]}
            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)

        for i in range(self.numStates2Check):
            state = self.state2Monitor[i,:]
            print(prevVal[i], "-->", list(self.actionValuesVec(state)))     

    def save_network(self):
        self.saver.save(self.sess, self.directoryName)
        print("save nn with", self.numRuns.eval(session = self.sess))

    def end_run(self):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)

class DoubleDQN:
    def __init__(self, modelParams, nnDirectory, learning_rate = 0.01):
        
        #net scopes 
        self.targetScope = nnDirectory + "_target"
        self.targetParamsName = 't'
        self.evalScope = nnDirectory + "_eval"
        self.evalParamsName = 'e'

        # Parameters
        self.learning_rate = learning_rate
        self.params = modelParams

        # Network Parameters
        self.num_input = modelParams.stateSize
        self.numActions = modelParams.numActions

        self.s = tf.placeholder(tf.float32, [None, self.num_input], name = 's') 
        self.s_ = tf.placeholder(tf.float32, [None, self.num_input], name = 's_') 
        self.r = tf.placeholder(tf.float32, [None, ], name='r') # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a') # input Action
        self.notDone = tf.placeholder(tf.float32, [None, ], name='notDone') # input Action

        self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer())
        #build double networks
        self.build_net(modelParams.nn_Func)

        # create operation to copy networks
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.targetScope)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.evalScope)
    
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if modelParams.outputGraph:
            # $ tensorboard --logdir=$(FName)_logs
            tf.summary.FileWriter(nnDirectory + "_logs/", self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())        
        
        # load
        self.directoryName = "./" + nnDirectory
        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta):
            tf.reset_default_graph()
            if "newDQN" not in sys.argv:
                print("\n\nload network\n\n")
                self.saver.restore(self.sess, self.directoryName)
            else:
                print("\n\nnew network\n\n")

        self.learnStepCounter = 0

        self.numStates2Check = 5
        self.state2Monitor = np.zeros((self.numStates2Check, modelParams.stateSize), dtype=np.int32, order='C')
        idxSelfLoc = [1, 25, 35, 96, 70]
        idxBeaconLoc = [0, 15, 0, 86, 10]
        for i in range(self.numStates2Check):
            self.state2Monitor[i,idxSelfLoc[i]] = 1
            self.state2Monitor[i,idxBeaconLoc[i]] = 1
        
    # Define the neural network
    def build_net(self, NN_Func):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        # Construct network
        self.targetNN = NN_Func(self.s_, self.numActions, w_initializer, b_initializer, self.targetScope, self.targetParamsName)
        self.evalNN = NN_Func(self.s, self.numActions, w_initializer, b_initializer, self.evalScope, self.evalParamsName)

        # compute R = r + t * d * 
        with tf.variable_scope(self.targetScope):
            q_target = self.r + self.params.discountFactor * self.notDone * tf.reduce_max(self.targetNN, axis=1, name='Qmax_s_')#self.notDone * 
            self.q_target = tf.stop_gradient(q_target)
        
        # select single action for evaluation
        with tf.variable_scope(self.evalScope):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.evalNN, indices=a_indices)

        # Define loss and train operation
        self.loss_op = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='loss_func'))
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_op, name="train_func")

    def choose_action(self, observation):
        if np.random.uniform() > self.params.ExplorationProb(self.numRuns.eval(session = self.sess)):
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.targetNN, feed_dict={self.s_: observation.reshape(1,self.num_input)})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.numActions)

        return action

    def TargetOut(self, state):
        allVals = self.targetNN.eval({self.s_: state.reshape(1,self.num_input)}, session=self.sess)
        return allVals[0]
    
    def EvalOut(self, state):
        allVals = self.evalNN.eval({self.s: state.reshape(1,self.num_input)}, session=self.sess)
        return allVals[0]
        
    def learn(self, s, a, r, s_, terminal):

        if (self.learnStepCounter % self.params.copyEvalToTarget) == 0:
            prevVal = []
            evalVal = []
            for i in range(self.numStates2Check):
                state = self.state2Monitor[i,:]
                prevVal.append(list(self.TargetOut(state))) 
                evalVal.append(list(self.EvalOut(state))) 
            self.sess.run(self.target_replace_op)
            updatedVal = []
            for i in range(self.numStates2Check):
                state = self.state2Monitor[i,:]
                updatedVal.append(list(self.TargetOut(state))) 
            print('\n\ntarget_params_replaced\n')
            for i in range(self.numStates2Check):
                print(prevVal[i], "-->", updatedVal[i], 'eval:', evalVal[i])               
        
        size = len(a)
        idxArray = np.arange(size)
        notDoneFloat = np.invert(terminal).astype(float)
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.random.choice(idxArray, self.params.batchSize)
            feedDict = {self.s: s[chosen].reshape(self.params.batchSize, self.num_input), 
                        self.s_: s_[chosen].reshape(self.params.batchSize, self.num_input),
                        self.r: r[chosen], self.a: a[chosen], self.notDone: notDoneFloat[chosen]}
            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)

        self.learnStepCounter += 1
        

    def save_network(self):
        self.saver.save(self.sess, self.directoryName)
        print("save nn with", self.numRuns.eval(session = self.sess))

    def end_run(self):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)        
        
        
        
        
        
        
        
        
        
        
        # prevVal = []
        # for i in range(self.numStates2Check):
        #     state = self.state2Monitor[i,:]
        #     prevVal.append(list(self.actionValuesVec(state))) 


        # for i in range(self.numStates2Check):
        #     state = self.state2Monitor[i,:]
        #     print(prevVal[i], "-->", list(self.actionValuesVec(state)))     