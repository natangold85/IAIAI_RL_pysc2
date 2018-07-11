import numpy as np
from scipy import sparse

import pandas as pd
import random
import pickle
import os.path

import tensorflow as tf
import os

class DTN:
    def __init__(self, modelParams, nnDirectory, loadNN, learning_rate = 0.001):
        # Parameters
        self.params = modelParams
        self.learning_rate = learning_rate

        # Network Parameters
        self.num_output = modelParams.outputEnd - modelParams.outputStart 

        self.numRuns =tf.get_variable(nnDirectory + ".numRuns", shape=(), initializer=tf.zeros_initializer())
        
        self.inputLayerState = tf.placeholder(tf.float32, [None, modelParams.stateSize]) 
        self.inputLayerActions = tf.placeholder(tf.float32, [None, modelParams.numActions]) 

        # Construct network
        self.outputTensor = tf.placeholder(tf.float32, [None, self.num_output])
        self.outputLayer = self.build_dtn(modelParams.nn_Func, nnDirectory)

        # Define loss and optimizer
        self.loss_op = tf.squared_difference(self.outputLayer, self.outputTensor)
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
            if loadNN:
                self.saver.restore(self.sess, self.directoryName)

    # Define the neural network
    def build_dtn(self, NN_Func, scope):
        with tf.variable_scope(scope):
            inputLayer = tf.concat([self.inputLayerState, self.inputLayerActions], 1)
            fc1 = tf.contrib.layers.fully_connected(inputLayer, 512)
            output = tf.contrib.layers.fully_connected(fc1, self.num_output)
            outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

        return outputSoftmax

    def RegularizationFactor(self):
        return 0
      
    def predict(self, observation, action):
        normalizeTo = np.sum(observation[self.params.outputStart:self.params.outputEnd])
        actionVec = np.zeros((1,self.params.numActions), int)
        actionVec[0,action] = 1
        vals = self.outputLayer.eval({self.inputLayerState: observation.reshape(1, self.params.stateSize), self.inputLayerActions: actionVec}, session=self.sess)

        return vals * normalizeTo
        
    def learn(self, s, a, s_):             
        size = len(a)
        # construct sparse repressentation of actions
        actionIdx = np.arange(size)
        actionInput=sparse.csc_matrix((a,(actionIdx,actionIdx)),shape=(size,self.params.numActions))

        idxArray = np.arange(size)
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.random.choice(idxArray, self.params.batchSize)
            feedDict = {self.inputLayerState: s[chosen].reshape(self.params.batchSize, self.params.stateSize), self.inputLayerActions: actionInput[chosen,:]}

            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)
  

    def save_network(self):
        self.saver.save(self.sess, self.directoryName)
        print("save nn with", self.numRuns.eval(session = self.sess))

    def NumRuns(self):
        return int(self.numRuns.eval(session = self.sess))

    def end_run(self):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)

class DoubleDQN:
    def __init__(self, modelParams, nnDirectory, loadNN, learning_rate = 0.01):
        
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
            if loadNN:
                self.saver.restore(self.sess, self.directoryName)

        self.learnStepCounter = 0

        self.numStates2Check = len(self.params.state2Monitor)
        
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
        if np.random.uniform() > self.params.ExploreProb(self.numRuns.eval(session = self.sess)):
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
                state = self.params.state2Monitor[i]
                prevVal.append(list(self.TargetOut(state))) 
            self.sess.run(self.target_replace_op)
            updatedVal = []

            for i in range(self.numStates2Check):
                state = self.params.state2Monitor[i]
                updatedVal.append(list(self.TargetOut(state))) 
            print('\n\ntarget_params_replaced\n')
            
            for i in range(self.numStates2Check):
                print(prevVal[i], "-->", updatedVal[i])               
        
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