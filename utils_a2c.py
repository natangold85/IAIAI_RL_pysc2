import numpy as np
import pandas as pd
import random
import pickle
import os.path
import threading

import tensorflow as tf
import os

from utils import ParamsBase

from multiprocessing import Lock
from utils import EmptyLock

# dqn params
class A2C_PARAMS(ParamsBase):
    def __init__(self, stateSize, numActions, discountFactor = 0.95, batchSize = 32, maxReplaySize = 500000, minReplaySize = 1000, 
                learning_rate=0.00001, numTrials2CmpResults=1000):

        super(DQN_PARAMS, self).__init__(stateSize=stateSize, numActions=numActions, discountFactor=discountFactor, 
                                            maxReplaySize=maxReplaySize, minReplaySize=minReplaySize)
        

        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.type = "A3C"

        self.numTrials2CmpResults = numTrials2CmpResults


class A3C:
    def __init__(self, modelParams, nnName, nnDirectory, loadNN, isMultiThreaded = False, agentName = "", createSaver = True):
        # Parameters
        self.params = modelParams

        # Network Parameters
        self.num_input = modelParams.stateSize
        self.numActions = modelParams.numActions

        self.directoryName = nnDirectory + nnName
        self.agentName = agentName
        
        self.scope = self.params.scopeVarName if self.params.scopeVarName != '' else self.directoryName

        with tf.variable_scope(self.scope):
            self.state = tf.placeholder(tf.float32, shape=[None, self.num_input], name="state")  # (None, 84, 84, 4)
            self.action = tf.placeholder(tf.float32, shape=[None, self.numActions], name="action")  # (None, actions)
            self.target_q = tf.placeholder(tf.float32, shape=[None])

            self.outputLayer = self.build_a3c(modelParams.nn_Func, self.scope)

        with tf.variable_scope(self.scope):
            batch_size = tf.shape(self.inputLayer)[0]

            gather_indices = tf.range(batch_size) * tf.shape(self.outputLayer)[1] + self.actionSelected
            action_predictions = tf.gather(tf.reshape(self.outputLayer, [-1]), gather_indices, name="action_selected_value")
            boundedPrediction = tf.clip_by_value(action_predictions, -1.0, 1.0, name="clipped_action_selected_value")

            # Define loss and optimizer
            lossFunc = tf.squared_difference(self.outputSingle, boundedPrediction)
            self.loss_op = tf.reduce_mean(lossFunc + self.RegularizationFactor(), name="loss_func")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op, name="train_func")


        if modelParams.tfSession == None:
            self.sess = tf.Session()    
            if modelParams.outputGraph:
                # $ tensorboard --logdir=logs
                tf.summary.FileWriter(nnDirectory + "/", self.sess.graph)
        
        else: 
            self.sess = modelParams.tfSession
        
        
        # Initializing session and variables
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

        if createSaver:
            self.saver = tf.train.Saver()
            fnameNNMeta = self.directoryName + ".meta"
            if os.path.isfile(fnameNNMeta) and loadNN:
                self.saver.restore(self.sess, self.directoryName)
            else:
                self.SaveDQN()
        else:
            self.saver = None

    # Define the neural network
    def build_a3c(self, NN_Func, scope):
        if NN_Func != None:
            return NN_Func(self.inputLayer, self.numActions, scope)   

        with tf.variable_scope(scope):
            currInput = self.inputLayer
            for i in range(self.params.layersNum):
                fc = tf.contrib.layers.fully_connected(currInput, self.params.neuronsInLayerNum)
                currInput = fc

            output = tf.contrib.layers.fully_connected(currInput, self.numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
            
        return output

    def RegularizationFactor(self):
        return 0

    def ExploreProb(self):
        return self.params.ExploreProb(self.numRuns.eval(session = self.sess))

    def TargetExploreProb(self):
        return self.ExploreProb()    

    def choose_action(self, observation):
        if np.random.uniform() > self.params.ExploreProb(self.numRuns.eval(session = self.sess)):
            vals = self.outputLayer.eval({self.inputLayer: observation.reshape(1,self.num_input)}, session=self.sess)

            maxArgs = np.argwhere(vals[0] == np.amax(vals[0]))
            a = np.random.choice(maxArgs[0])      
        else:
            a = np.random.randint(0, self.numActions)

        return a


    def ActionValuesVec(self, state, targetValues = False):
        allVals = self.outputLayer.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess)
        return allVals[0]

    def learn(self, s, a, r, s_, terminal, numRuns2Save = None):          
        size = len(a)

        if self.params.noiseOnTerminalRewardsPct > 0:
            r = self.NoiseOnTerminalReward(r, terminal)
        
        # calculate (R = r + d * Q(s_))
        rNextState = self.outputLayer.eval({self.inputLayer: s_.reshape(size,self.num_input)}, session=self.sess)
        R = r + np.invert(terminal) * self.params.discountFactor * np.max(rNextState, axis=1)

        for i in range(int(size  / self.params.batchSize)):
            chosen = np.arange(i * self.params.batchSize, (i + 1) * self.params.batchSize)
            feedDict = {self.inputLayer: s[chosen].reshape(self.params.batchSize, self.num_input), self.outputSingle: R[chosen], self.actionSelected: a[chosen]}

            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict) 


    def NoiseOnTerminalReward(self, r, terminal):
        idxTerminal = np.argwhere(terminal).flatten()
        sizeTerminal = len(idxTerminal)
        sizeNoise = int(sizeTerminal * self.params.noiseOnTerminalRewardsPct)
        idxNoise = np.random.choice(idxTerminal, sizeNoise)
        r[idxNoise] *= -1

        return r
        
    def Close(self):
        self.sess.close()
        
    def SaveDQN(self, numRuns2Save = None, toPrint = True):
        
        if numRuns2Save == None:
            self.saver.save(self.sess, self.directoryName)
            numRuns2Save = self.NumRuns()
        else:
            currNumRuns = self.NumRuns()
            assign4Save = self.numRuns.assign(numRuns2Save)
            self.sess.run(assign4Save)

            self.saver.save(self.sess, self.directoryName)

            assign4Repeat2Train = self.numRuns.assign(currNumRuns)
            self.sess.run(assign4Repeat2Train)

        if toPrint:
            print("\n\t", threading.current_thread().getName(), " : ", self.agentName, "->save dqn with", numRuns2Save, "runs")

    def NumRuns(self):
        return self.numRuns.eval(session = self.sess)

    def end_run(self, reward, currentNumRuns):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)
        
    def DiscountFactor(self):
        return self.params.discountFactor

    def Reset(self):
        self.sess.run(self.init_op) 
    
    def IsWithDfltDecisionMaker(self):
        return False

    def NumDfltRuns(self):
        return 0

    def TakeDfltValues(self):
        return False

    def GetAllNNVars(self):
        nnVars = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        nnVars = sorted(nnVars, key=lambda v: v.name)

        npVars = []
        varName = []
        for v in range(len(nnVars)):
            varName.append(nnVars[v].name)
            npVars.append(nnVars[v].eval(session = self.sess))

        return npVars, varName

    def AssignAllNNVars(self, newValues):
        nnVars = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        nnVars = sorted(nnVars, key=lambda v: v.name)

        copy_ops = []
        for v in range(len(nnVars)):
            op = nnVars[v].assign(newValues[v])
            copy_ops.append(op)

        self.sess.run(copy_ops)

    def actionValuesSpecific(self, state, dmId): # dmId = target, curr
        isTarget = dmId == "target"
        return self.ActionValuesVec(state, isTarget)
