import numpy as np
import pandas as pd
import random
import pickle
import os.path
import threading

import tensorflow as tf
import os

from utils import ParamsBase

# dqn params
class DQN_PARAMS(ParamsBase):
    def __init__(self, stateSize, numActions, layersNum = 1, neuronsInLayerNum = 256, numTrials2CmpResults = 1000, historyProportion4Learn = 1, nn_Func = None, propogateReward = False, outputGraph = False, discountFactor = 0.95, batchSize = 32, maxReplaySize = 500000, minReplaySize = 1000, copyEvalToTarget = 5, explorationProb = 0.1, descendingExploration = True, exploreChangeRate = 0.001, states2Monitor = [], scopeVarName = '', normalizeRewards = True):
        super(DQN_PARAMS, self).__init__(stateSize, numActions, historyProportion4Learn, propogateReward, discountFactor, maxReplaySize, minReplaySize)
        
        self.nn_Func = nn_Func
        self.batchSize = batchSize

        self.outputGraph = outputGraph
        
        self.copyEvalToTarget = copyEvalToTarget

        self.explorationProb = explorationProb
        self.descendingExploration = descendingExploration
        self.exploreChangeRate = exploreChangeRate 

        self.type = "DQN"
        self.scopeVarName = scopeVarName
        self.tfSession = None

        self.numTrials2CmpResults = numTrials2CmpResults

        self.layersNum = layersNum
        self.neuronsInLayerNum = neuronsInLayerNum

        self.normalizeRewards = normalizeRewards

    def ExploreProb(self, numRuns, resultRatio = 1):
        if self.descendingExploration:
            return self.explorationProb + (1 - self.explorationProb) * np.exp(-self.exploreChangeRate * resultRatio * numRuns)
        else:
            return self.explorationProb

class DQN_EMBEDDING_PARAMS(DQN_PARAMS):
    def __init__(self, stateSize, embeddingInputSize, numActions, numTrials2CmpResults = 1000, historyProportion4Learn = 1, nn_Func = None, propogateReward = False, outputGraph = False, discountFactor = 0.95, batchSize = 32, maxReplaySize = 500000, minReplaySize = 1000, copyEvalToTarget = 5, explorationProb = 0.1, descendingExploration = True, exploreChangeRate = 0.0005, states2Monitor = [], scopeVarName = ''):
        super(DQN_EMBEDDING_PARAMS, self).__init__(stateSize, numActions, numTrials2CmpResults, historyProportion4Learn, nn_Func, propogateReward, outputGraph, discountFactor, batchSize, maxReplaySize, minReplaySize, copyEvalToTarget, explorationProb, descendingExploration, exploreChangeRate, states2Monitor, scopeVarName)
        
        self.embeddingInputSize = embeddingInputSize
        self.type = "DQN_Embedding"
        


class DQN:
    def __init__(self, modelParams, nnName, nnDirectory, loadNN, isMultiThreaded = False, createSaver = True, learning_rate = 0.00001):
        # Parameters
        self.params = modelParams
        self.learning_rate = learning_rate

        # Network Parameters
        self.num_input = modelParams.stateSize
        self.numActions = modelParams.numActions

        self.directoryName = nnDirectory + nnName

        if self.params.scopeVarName == '':
            self.scope = nnName
        else:
            self.scope = self.params.scopeVarName

        with tf.variable_scope(self.scope):
            self.inputLayer = tf.placeholder("float", [None, self.num_input]) 
 
            self.outputSingle = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
            self.actionSelected = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer())
    
        # Construct network
        if modelParams.type == "DQN_Embedding":
            self.outputLayer = self.build_dqn_withEmbedding(modelParams.nn_Func, self.scope)
        else:
            self.outputLayer = self.build_dqn(modelParams.nn_Func, self.scope)

        with tf.variable_scope(self.scope):
            batch_size = tf.shape(self.inputLayer)[0]

            gather_indices = tf.range(batch_size) * tf.shape(self.outputLayer)[1] + self.actionSelected
            action_predictions = tf.gather(tf.reshape(self.outputLayer, [-1]), gather_indices, name="action_selected_value")
            boundedPrediction = tf.clip_by_value(action_predictions, -1.0, 1.0, name="clipped_action_selected_value")

            # Define loss and optimizer
            lossFunc = tf.squared_difference(self.outputSingle, boundedPrediction)
            self.loss_op = tf.reduce_mean(lossFunc + self.RegularizationFactor(), name="loss_func")

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

        self.numStates2Check = len(self.params.states2Monitor)

    # Define the neural network
    def build_dqn(self, NN_Func, scope):
        if NN_Func != None:
            return NN_Func(self.inputLayer, self.numActions, scope)   

        with tf.variable_scope(scope):
            currInput = self.inputLayer
            for i in range(self.params.layersNum):
                fc = tf.contrib.layers.fully_connected(currInput, self.params.neuronsInLayerNum)
                currInput = fc

            output = tf.contrib.layers.fully_connected(currInput, self.numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
            
        return output

    # Define the neural network
    def build_dqn_withEmbedding(self, NN_Func, scope):
        
        with tf.variable_scope(scope):
            embedSize = self.params.embeddingInputSize
            restSize = self.params.stateSize - embedSize
            
            embeddingInput = tf.slice(self.inputLayer, [0,0], [-1,embedSize])
            otherInput = tf.slice(self.inputLayer, [0,embedSize], [-1,restSize])
        
        if NN_Func != None:
            return NN_Func(embeddingInput, otherInput, self.numActions, scope)   


        with tf.variable_scope(scope):
            embeddingOut = tf.contrib.layers.fully_connected(embeddingInput, 256)
            currInput = tf.concat([embeddingOut, otherInput], axis = 1)

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

    def learn(self, s, a, r, s_, terminal):          
        size = len(a)
        
        # calculate (R = r + d * Q(s_))
        rNextState = self.outputLayer.eval({self.inputLayer: s_.reshape(size,self.num_input)}, session=self.sess)
        R = r + np.invert(terminal) * self.params.discountFactor * np.max(rNextState, axis=1)

        # feedDictAll = {self.inputLayer: s.reshape(size, self.num_input), self.outputSingle: R, self.actionSelected: a}
        # lossBefore = self.sess.run([self.loss_op], feed_dict=feedDictAll)  
        for i in range(int(size  / self.params.batchSize)):
            chosen = np.arange(i * self.params.batchSize, (i + 1) * self.params.batchSize)
            feedDict = {self.inputLayer: s[chosen].reshape(self.params.batchSize, self.num_input), self.outputSingle: R[chosen], self.actionSelected: a[chosen]}

            self.sess.run([self.train_op, self.loss_op], feed_dict=feedDict) 

        # lossAfter = self.sess.run([self.loss_op], feed_dict=feedDictAll)    
        
        # rNextStateAfter = self.outputLayer.eval({self.inputLayer: s_.reshape(size,self.num_input)}, session=self.sess)
        # R_after = r + np.invert(terminal) * self.params.discountFactor * np.max(rNextStateAfter, axis=1)
        # feedDictAfter = {self.inputLayer: s.reshape(size, self.num_input), self.outputSingle: R_after, self.actionSelected: a}
        # lossAfterWithR = self.sess.run([self.loss_op], feed_dict=feedDictAfter)    
        # print("loss before =", '%f' % lossBefore[0], "loss after =", '%f' % lossAfter[0], "loss after on current r =", '%f' % lossAfterWithR[0])

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
            print("\n\t", threading.current_thread().getName(), " : save dqn with", numRuns2Save, "runs")

    def NumRuns(self):
        return int(self.numRuns.eval(session = self.sess))

    def end_run(self, reward, toSave = False):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)
        
    def DiscountFactor(self):
        return self.params.discountFactor

    def Reset(self):
        self.sess.run(self.init_op) 


class DQN_WithTarget(DQN):
    def __init__(self, modelParams, nnName, nnDirectory, loadNN, isMultiThreaded = False, learning_rate = 0.00001):
        super(DQN_WithTarget, self).__init__(modelParams=modelParams, nnName=nnName, nnDirectory=nnDirectory, isMultiThreaded=isMultiThreaded,
                                                loadNN=loadNN, learning_rate=learning_rate, createSaver=False)
        self.numTrials2CmpResults = modelParams.numTrials2CmpResults

        self.targetScope = self.scope + "_target"

        with tf.variable_scope(self.targetScope):
            self.numRunsTarget =tf.get_variable("numRuns_target", shape=(), initializer=tf.zeros_initializer())

        # Construct target network
        if modelParams.type == "DQN_Embedding":
            self.targetOutput = self.build_dqn_withEmbedding(modelParams.nn_Func, self.targetScope)
        else:
            self.targetOutput = self.build_dqn(modelParams.nn_Func, self.targetScope)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta) and loadNN:
            self.saver.restore(self.sess, self.directoryName)
        else:
            self.SaveDQN()

        if modelParams.outputGraph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter(nnDirectory + "/", self.sess.graph)

        self.rewardHist = []
        self.bestReward = -1000

    def CopyDqn2Target(self):
        dqnParams = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        dqnParams = sorted(dqnParams, key=lambda v: v.name)

        targetParams = [t for t in tf.trainable_variables() if t.name.startswith(self.targetScope)]
        targetParams = sorted(targetParams, key=lambda v: v.name)

        update_ops = []
        for dqnVar, targetVar in zip(dqnParams, targetParams):
            op = targetVar.assign(dqnVar)
            update_ops.append(op)

        self.sess.run(update_ops)
        self.SaveDQN(toPrint=False)

    def CopyTarget2DQN(self):
        numRuns = self.NumRuns()
        
        dqnParams = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        dqnParams = sorted(dqnParams, key=lambda v: v.name)
 
        targetParams = [t for t in tf.trainable_variables() if t.name.startswith(self.targetScope)]
        targetParams = sorted(targetParams, key=lambda v: v.name)

        update_ops = []
        for dqnVar, targetVar in zip(dqnParams, targetParams):
            op = dqnVar.assign(targetVar)
            update_ops.append(op)

        self.sess.run(update_ops)
        assign = self.numRuns.assign(numRuns)
        self.sess.run(assign)
        self.SaveDQN()


    def ActionValuesVec(self, state, targetValues = False):
        if targetValues:
            allVals = self.targetOutput.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess)
            return allVals[0]
        else:
            return super(DQN_WithTarget, self).ActionValuesVec(state, targetValues)


    def TargetExploreProb(self):
        return 0
        
    def NumRunsTarget(self):
        return int(self.numRunsTarget.eval(session = self.sess))

    def end_run(self, r, toSave = False):
        super(DQN_WithTarget, self).end_run(r, toSave)

        # insert reward to reward history and pop first from histor if necessary
        self.rewardHist.append(r)
        if len(self.rewardHist) > self.numTrials2CmpResults:
            self.rewardHist.pop(0)
        
        # calculate results and compare to target
        if len(self.rewardHist) == self.numTrials2CmpResults:
            avgReward = np.average(np.array(self.rewardHist))
            if avgReward > self.bestReward:
                self.bestReward = avgReward
                self.CopyDqn2Target()
