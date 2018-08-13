import numpy as np
from scipy import sparse

import datetime
import pandas as pd
import random
import pickle
import os.path

import tensorflow as tf
import os

class DTN_PARAMS:
    def __init__(self, stateSize, numActions, outputStart, outputEnd, nn_Func = None, outputGraph = False, batchSize = 32, minReplaySize = 100, maxReplaySize = 500000):
        self.stateSize = stateSize
        self.numActions = numActions
        self.outputStart = outputStart
        self.outputEnd = outputEnd
        self.nn_Func = nn_Func
        self.outputGraph = outputGraph
        self.batchSize = batchSize

        self.minReplaySize = minReplaySize
        self.maxReplaySize = maxReplaySize

class DTN:
    def __init__(self, modelParams, nnName, directory, loadNN = True, learning_rate = 0.001):

        self.hist = {}
        self.hist["s"] = []
        self.hist["s_"] = []
        self.hist["a"] = []


        # Parameters
        self.params = modelParams
        self.learning_rate = learning_rate

        self.nnName = nnName
        self.directoryName = directory + nnName

        # Network Parameters
        self.num_output = modelParams.outputEnd - modelParams.outputStart 

        self.numRuns =tf.get_variable(nnName + ".numRuns", shape=(), initializer=tf.zeros_initializer())
        
        self.inputLayerState = tf.placeholder(tf.float32, [None, modelParams.stateSize]) 
        self.inputLayerActions = tf.placeholder(tf.float32, [None, modelParams.numActions]) 

        # Construct network
        self.outputTensor = tf.placeholder(tf.float32, [None, self.num_output])
        self.outputLayer = self.build_dtn(modelParams.nn_Func, nnName)

        # Define loss and optimizer
        self.loss_op = tf.squared_difference(self.outputLayer, self.outputTensor)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name="train_func")

        # Initializing session and variables

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        if modelParams.outputGraph:
            # $ tensorboard --logdir=directory
            tf.summary.FileWriter(directory + "/", self.sess.graph)

        self.sess.run(self.init_op) 

        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta):
            if loadNN:
                self.saver.restore(self.sess, self.directoryName)

        
        self.lastLearningTime = 0
        self.lastSavingTime = 0

    # Define the neural network
    def build_dtn(self, NN_Func, scope):
        if NN_Func != None:
            return NN_Func(self.inputLayerState, self.inputLayerActions, self.num_output, scope)

        with tf.variable_scope(scope):

            el1 = tf.contrib.layers.fully_connected(self.inputLayerState, 256)
            middleLayer = tf.concat([el1, self.inputLayerActions], 1)
            fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
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
        start = datetime.datetime.now()           
        size = len(a)
        # construct sparse repressentation of actions
        idxArray = np.arange(size)
        valInSparse = np.ones(size, dtype = int)
        actionInput=sparse.csc_matrix((valInSparse,(idxArray,a)),shape=(size,self.params.numActions)).toarray()
        
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.random.choice(idxArray, self.params.batchSize)
            
            sChosen = s[chosen].reshape(self.params.batchSize, self.params.stateSize)
            s_Chosen = s_[chosen].reshape(self.params.batchSize, self.params.stateSize)
            aChosen = actionInput[chosen,:]
            
            feedDict = {self.inputLayerState: sChosen, self.inputLayerActions: aChosen, self.outputTensor: s_Chosen}
            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)

        diff = datetime.datetime.now() - start
        self.lastLearningTime = diff.seconds * 1000 + diff.microseconds / 1000


    def Reset(self):
        #self.outputLayer = self.build_dtn(self.params.nn_Func, self.nnName)
        # tf.reset_default_graph()
        self.sess.run(self.init_op) 
        assign_op = self.numRuns.assign(0)
        self.sess.run(assign_op)

        self.hist = {}
        self.hist["s"] = []
        self.hist["s_"] = []
        self.hist["a"] = []


    def save_network(self):
        start = datetime.datetime.now() 
        self.saver.save(self.sess, self.directoryName)
        diff = datetime.datetime.now() - start
        self.lastSavingTime = diff.seconds * 1000 + diff.microseconds / 1000
        print("save", self.directoryName, "with", self.numRuns.eval(session = self.sess), "runs", "save duration =", self.lastSavingTime)

    def NewExperience(self, s, a, s_):
        self.hist["s"].append(s)
        self.hist["a"].append(a)
        self.hist["s_"].append(s_)

    def GetHist(self):
        while len(self.hist["a"]) > self.params.maxReplaySize:
            self.hist["s"].pop(0)
            self.hist["a"].pop(0)
            self.hist["s_"].pop(0)

        size = len(self.hist["a"])
        idxShuffle = np.arange(size)
        np.random.shuffle(idxShuffle)

        s = np.array(self.hist["s"])[idxShuffle]
        a = np.array(self.hist["a"])[idxShuffle]
        s_ = np.array(self.hist["s_"])[idxShuffle]

        return s, a, s_

    def NumRuns(self):
        return int(self.numRuns.eval(session = self.sess))

    def end_run(self, toLearn = False, toSave = False, numRuns = 1):
        assign = self.numRuns.assign_add(numRuns)
        self.sess.run(assign)

        s, a, s_ = self.GetHist()
        if len(a) >= self.params.minReplaySize:
            print("DTN - start training with hist size =", len(a), end = ', ')   
            self.learn(s, a, s_)
            print("end taining after", self.lastLearningTime, "ms")

        if toSave:
            self.save_network()

    def LastTrainDuration(self):
        return self.lastLearningTime
    def LastSaveDuration(self):
        return self.lastSavingTime





class DTN2:
    def __init__(self, modelParams, nnName, directory, loadNN = True, learning_rate = 0.001):
        self.hist = {}
        self.hist["s"] = []
        self.hist["s_"] = []
        self.hist["a"] = []
        self.hist["p"] = []

        # Parameters
        self.params = modelParams
        self.learning_rate = learning_rate

        self.nnName = nnName
        self.directoryName = directory + nnName

        # Network Parameters
        self.num_output = 1 

        self.numRuns =tf.get_variable(nnName + ".numRuns", shape=(), initializer=tf.zeros_initializer())
        
        self.inputLayerState = tf.placeholder(tf.float32, [None, modelParams.stateSize]) 
        self.inputLayerNextState = tf.placeholder(tf.float32, [None, modelParams.stateSize]) 
        self.inputLayerActions = tf.placeholder(tf.float32, [None, modelParams.numActions]) 

        # Construct network
        self.outputTensor = tf.placeholder(tf.float32, [None, self.num_output])
        self.outputLayer = self.build_dtn(modelParams.nn_Func, nnName)

        # Define loss and optimizer
        self.loss_op = tf.squared_difference(self.outputLayer, self.outputTensor)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name="train_func")

        # Initializing session and variables

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        if modelParams.outputGraph:
            # $ tensorboard --logdir=directory
            tf.summary.FileWriter(directory + "/", self.sess.graph)

        self.sess.run(self.init_op) 

        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta):
            if loadNN:
                self.saver.restore(self.sess, self.directoryName)

        
        self.lastLearningTime = 0
        self.lastSavingTime = 0

    # Define the neural network
    def build_dtn(self, NN_Func, scope):
        if NN_Func != None:
            return NN_Func(self.inputLayerState, self.inputLayerNextState, self.inputLayerActions, self.num_output, scope)

        with tf.variable_scope(scope):

            el1_s = tf.contrib.layers.fully_connected(self.inputLayerState, 256)
            el1_s_ = tf.contrib.layers.fully_connected(self.inputLayerNextState, 256)
            middleLayer = tf.concat([el1_s, el1_s_, self.inputLayerActions], 1)
            fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
            output = tf.contrib.layers.fully_connected(fc1, self.num_output)

        return output

    def RegularizationFactor(self):
        return 0
    
    def TransitionProb(self, s, a, s_):
        actionVec = np.zeros((1,self.params.numActions), int)
        actionVec[0, a] = 1
        inputDict = {self.inputLayerState: s.reshape(1, self.params.stateSize), self.inputLayerNextState: s_.reshape(1, self.params.stateSize), 
                        self.inputLayerActions: actionVec}
        
        prob = self.outputLayer.eval(inputDict, session=self.sess)
        
        return prob[0]
    
    def NewExperience(self, s, a, s_, p):
        self.hist["s"].append(s)
        self.hist["a"].append(a)
        self.hist["s_"].append(s_)
        self.hist["p"].append(p)
    
    def learn(self, s, a, s_, p):  
        start = datetime.datetime.now()           
        size = len(a)
        # construct sparse repressentation of actions
        idxArray = np.arange(size)
        valInSparse = np.ones(size, dtype = int)
        actionInput=sparse.csc_matrix((valInSparse,(idxArray,a)),shape=(size,self.params.numActions)).toarray()
        
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.random.choice(idxArray, self.params.batchSize)
            
            sChosen = s[chosen].reshape(self.params.batchSize, self.params.stateSize)
            s_Chosen = s_[chosen].reshape(self.params.batchSize, self.params.stateSize)
            aChosen = actionInput[chosen,:]
            pChosen = p[chosen]
            
            feedDict = {self.inputLayerState: sChosen, 
                        self.inputLayerNextState: s_Chosen, 
                        self.inputLayerActions: aChosen, 
                        self.outputTensor: pChosen.reshape(self.params.batchSize, 1)}

            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict)

        diff = datetime.datetime.now() - start
        self.lastLearningTime = diff.seconds * 1000 + diff.microseconds / 1000


    def Reset(self):
        #self.outputLayer = self.build_dtn(self.params.nn_Func, self.nnName)
        #tf.reset_default_graph()
        self.sess.run(self.init_op) 
        assign_op = self.numRuns.assign(0)
        self.sess.run(assign_op)

        self.hist = {}
        self.hist["s"] = []
        self.hist["s_"] = []
        self.hist["a"] = []
        self.hist["p"] = []


    def save_network(self):
        start = datetime.datetime.now() 
        self.saver.save(self.sess, self.directoryName)
        diff = datetime.datetime.now() - start
        self.lastSavingTime = diff.seconds * 1000 + diff.microseconds / 1000
        print("save", self.directoryName, "with", self.numRuns.eval(session = self.sess), "runs", "save duration =", self.lastSavingTime)

    def NumRuns(self):
        return int(self.numRuns.eval(session = self.sess))

    def end_run(self, toLearn = False, toSave = True, numRuns = 1):
        assign = self.numRuns.assign_add(numRuns)
        self.sess.run(assign)

        s, a, s_, p = self.GetHist()
        if len(p) >= self.params.minReplaySize:
            print("DTN2 - start training with hist size =", len(p), end = ', ')
            self.learn(s, a, s_, p)
            print("end taining after", self.lastLearningTime, "ms")
        if toSave:
            self.save_network()

    def GetHist(self):
        while len(self.hist["a"]) > self.params.maxReplaySize:
            self.hist["s"].pop(0)
            self.hist["a"].pop(0)
            self.hist["s_"].pop(0)
            self.hist["p"].pop(0)
        
        size = len(self.hist["a"])
        idxShuffle = np.arange(size)
        np.random.shuffle(idxShuffle)

        s = np.array(self.hist["s"])[idxShuffle]
        a = np.array(self.hist["a"])[idxShuffle]
        s_ = np.array(self.hist["s_"])[idxShuffle]
        p = np.array(self.hist["p"])[idxShuffle]

        return s, a, s_, p

    def LastTrainDuration(self):
        return self.lastLearningTime
    def LastSaveDuration(self):
        return self.lastSavingTime



from utils_ttable import TransitionTable

class Filtered_DTN(DTN):
    def __init__(self, modelParams, nnName, directory, ttableName, loadNN = True, learning_rate = 0.001, thresholdNumTransitions = 1):
        super(Filtered_DTN, self).__init__(modelParams, nnName, directory, loadNN, learning_rate)
        
        self.ttableName = ttableName
        self.ttable = TransitionTable(modelParams.numActions, ttableName)
        self.thresholdNumTransitions = thresholdNumTransitions

    def learn(self, s, a, s_):
        for i in range(len(a)): 
            self.ttable.learn(str(s[i]), a[i], str(s_[i]))

        sLearn = []
        aLearn = []
        s_Learn = []

        allStates = self.ttable.table.keys()
        idxActionCount = self.ttable.actionSumIdx
        idxTable = self.ttable.tableIdx
        for sT in allStates:
            if sT == 'TrialsData':
                continue
            sStr = sT.replace("[", "").replace("]", "")
            sArray = np.fromstring(sStr, dtype=int, sep = ' ')
            actionCountVec = np.array(self.ttable.table[sT][idxActionCount])
            stateTable = self.ttable.table[sT][idxTable]
            thresholdActions = np.arange(len(actionCountVec))[actionCountVec > self.thresholdNumTransitions]

            for action in thresholdActions:
                sLearn.append(sArray)
                aLearn.append(action)
                s_Learn.append(self.CalcSPrime(stateTable, action))
        
        if len(aLearn) >= self.params.batchSize:
            super(Filtered_DTN, self).learn(np.array(sLearn), np.array(aLearn), np.array(s_Learn))
        else:
            print("\n\nskip learning")

    def CalcSPrime(self, stateTable, a):
        names = list(stateTable.index)

        sPrime = np.zeros(self.num_output)
        allCount = 0
        for sP in names:
            c = stateTable.ix[sP, a]
            if c > 0:
                sPArray = np.fromstring(sP.replace("[", "").replace("]", ""), dtype=int, sep = ' ')
                sPrime[sPArray > 0] = c
                allCount += c

        return sPrime / allCount

    def NewTTable(self):
        self.ttable = TransitionTable(self.params.numActions, self.ttableName)
            