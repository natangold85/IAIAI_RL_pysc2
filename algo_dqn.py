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
class DQN_PARAMS(ParamsBase):
    def __init__(self, stateSize, numActions, layersNum=1, neuronsInLayerNum=256, numTrials2CmpResults=1000, nn_Func=None, numTrials2Learn=None, numTrials2Save=100,
                outputGraph=True, discountFactor=0.95, batchSize=32, maxReplaySize=500000, minReplaySize=1000, 
                explorationProb=0.1, descendingExploration=True, exploreChangeRate=0.001, learning_rate=1e-05, 
                normalizeRewards=False, normalizeState=True, numRepeatsTerminalLearning=10, accumulateHistory=True):

        super(DQN_PARAMS, self).__init__(stateSize=stateSize, numActions=numActions, discountFactor=discountFactor, 
                                        maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, numTrials2Learn=numTrials2Learn, numTrials2Save=numTrials2Save)
        

        self.learning_rate = learning_rate
        self.nn_Func = nn_Func
        self.batchSize = batchSize

        self.outputGraph = outputGraph
        
        self.explorationProb = explorationProb
        self.descendingExploration = descendingExploration
        self.exploreChangeRate = exploreChangeRate 

        self.type = "DQN"
        self.tfSession = None

        self.numTrials2CmpResults = numTrials2CmpResults

        self.layersNum = layersNum
        self.neuronsInLayerNum = neuronsInLayerNum

        self.normalizeRewards = normalizeRewards
        self.normalizeState = normalizeState

        self.noiseOnTerminalRewardsPct = 0.0
        self.numRepeatsTerminalLearning = numRepeatsTerminalLearning

    def ExploreProb(self, numRuns, resultRatio = 1):
        if self.descendingExploration:
            return self.explorationProb + (1 - self.explorationProb) * np.exp(-self.exploreChangeRate * resultRatio * numRuns)
        else:
            return self.explorationProb

class DQN_EMBEDDING_PARAMS(DQN_PARAMS):
    def __init__(self, stateSize, embeddingInputSize, numActions, numTrials2CmpResults=1000, nn_Func=None, outputGraph=False, numTrials2Learn=None, numTrials2Save=100, 
                discountFactor=0.95, batchSize=32, maxReplaySize=500000, minReplaySize=1000, explorationProb=0.1, descendingExploration=True, 
                exploreChangeRate = 0.0005, accumulateHistory=True):
        
        super(DQN_EMBEDDING_PARAMS, self).__init__(stateSize=stateSize, numActions=numActions, numTrials2CmpResults=numTrials2CmpResults, nn_Func=nn_Func, 
                                                    outputGraph=outputGraph, discountFactor=discountFactor, batchSize=batchSize, maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, 
                                                    explorationProb=explorationProb, descendingExploration=descendingExploration, exploreChangeRate=exploreChangeRate, 
                                                    accumulateHistory=accumulateHistory, numTrials2Learn=numTrials2Learn, numTrials2Save=numTrials2Save)
        
        self.embeddingInputSize = embeddingInputSize
        self.type = "DQN_Embedding"
        
class DQN_PARAMS_WITH_DEFAULT_DM(DQN_PARAMS):
    def __init__(self, stateSize, numActions, numTrials2CmpResults = 1000, nn_Func = None, outputGraph = False, numTrials2Learn=None, numTrials2Save=100, 
                discountFactor = 0.95, batchSize = 32, maxReplaySize = 500000, minReplaySize = 1000, explorationProb = 0.1, descendingExploration = True, 
                exploreChangeRate = 0.0005, layersNum = 1, neuronsInLayerNum = 256, accumulateHistory=True):
        
        super(DQN_PARAMS_WITH_DEFAULT_DM, self).__init__(stateSize=stateSize, numActions=numActions, numTrials2CmpResults=numTrials2CmpResults, nn_Func=nn_Func, layersNum=layersNum, neuronsInLayerNum=neuronsInLayerNum,
                                                    outputGraph=outputGraph, discountFactor=discountFactor, batchSize=batchSize, maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, 
                                                    explorationProb=explorationProb, descendingExploration=descendingExploration, exploreChangeRate=exploreChangeRate, 
                                                    accumulateHistory=accumulateHistory, numTrials2Learn=numTrials2Learn, numTrials2Save=numTrials2Save)

        self.defaultDecisionMaker = None
        self.type = "DQN_With_Default"


class DQN:
    def __init__(self, modelParams, nnName, nnDirectory, isMultiThreaded = False, agentName = ""):
        # Parameters
        self.params = modelParams

        # Network Parameters
        self.num_input = modelParams.stateSize
        self.numActions = modelParams.numActions

        self.directory = nnDirectory
        self.nnName = nnName
        self.savePath = self.directory + self.nnName
        self.agentName = agentName

        self.dqnScope = "DQNetwork"
     
        self.inputLayer = tf.placeholder("float", [None, self.num_input], name="state") 
        self.actionSelected = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.outputSingle = tf.placeholder(shape=[None], dtype=tf.float32, name="value")
        
        with tf.variable_scope(self.dqnScope):
            self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32)

            # Construct network
            if modelParams.type == "DQN_Embedding":
                self.outputLayer = self.build_dqn_withEmbedding(modelParams.nn_Func)
            else:
                self.outputLayer = self.build_dqn(modelParams.nn_Func)

        with tf.variable_scope("action_prediction"):
            batch_size = tf.shape(self.actionSelected)[0]

            gather_indices = tf.range(batch_size) * tf.shape(self.outputLayer)[1] + self.actionSelected
            action_predictions = tf.gather(tf.reshape(self.outputLayer, [-1]), gather_indices, name="action_selected_value")
            boundedPrediction = tf.clip_by_value(action_predictions, -1.0, 1.0, name="clipped_action_selected_value")

        # Define loss and optimizer
        lossFunc = tf.squared_difference(self.outputSingle, boundedPrediction)
        self.loss_op = tf.reduce_mean(lossFunc + self.RegularizationFactor(), name="loss_func")

        optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name="train_func")
        
        
    def InitModel(self, session, resetModel=False):
        self.sess = session
        if self.params.outputGraph:
            # $ tensorboard --logdir=logs/nnDirectory
            tf.summary.FileWriter("./logs" + self.directory.replace(".", "") + "/", self.sess.graph)

        # Initializing session and variables
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

        self.saver = tf.train.Saver()
        fnameNNMeta = self.savePath + ".meta"
        if os.path.isfile(fnameNNMeta) and not resetModel:
            self.saver.restore(self.sess, self.savePath)
            loadedDM = True
        else:
            self.Save()
            loadedDM = False
        
        return loadedDM

    # Define the neural network
    def build_dqn(self, NN_Func):
        if NN_Func != None:
            return NN_Func(self.inputLayer, self.numActions)   

        # with tf.variable_scope(self.scope + "/" + scope):
        currInput = self.inputLayer
        for i in range(self.params.layersNum):
            fc = tf.contrib.layers.fully_connected(currInput, self.params.neuronsInLayerNum)
            currInput = fc

        output = tf.contrib.layers.fully_connected(currInput, self.numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
        
        return output

    # Define the neural network
    def build_dqn_withEmbedding(self, NN_Func):
        
        embedSize = self.params.embeddingInputSize
        restSize = self.params.stateSize - embedSize
        
        embeddingInput = tf.slice(self.inputLayer, [0,0], [-1,embedSize])
        otherInput = tf.slice(self.inputLayer, [0,embedSize], [-1,restSize])
        
        if NN_Func != None:
            return NN_Func(embeddingInput, otherInput, self.numActions)   


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

    def choose_action(self, state, validActions, targetValues=False):
        vals = self.outputLayer.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess).squeeze()
        
        if np.random.uniform() > self.params.ExploreProb(self.numRuns.eval(session = self.sess)):
            maxArgs = np.argwhere(vals == np.amax(vals[validActions]))
            maxArgsValid = np.array([x for x in maxArgs if x in validActions]).squeeze(axis=1)
            a = np.random.choice(maxArgsValid)      
        else:
            a = np.random.choice(validActions)

        return a, vals


    def ActionsValues(self, state, validActions, targetValues = False):
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
        
    def Save(self, numRuns2Save = None, toPrint = True):
        
        if numRuns2Save == None:
            self.saver.save(self.sess, self.savePath)
            numRuns2Save = self.NumRuns()
        else:
            currNumRuns = self.NumRuns()
            assign4Save = self.numRuns.assign(numRuns2Save)
            self.sess.run(assign4Save)

            self.saver.save(self.sess, self.savePath)

            assign4Repeat2Train = self.numRuns.assign(currNumRuns)
            self.sess.run(assign4Repeat2Train)

        if toPrint:
            print("\n\t", threading.current_thread().getName(), " : ", self.agentName, "->save dqn with", numRuns2Save, "runs to:", self.savePath)

    def NumRuns(self):
        return self.numRuns.eval(session = self.sess)

    def end_run(self, reward):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)
        
    def DiscountFactor(self):
        return self.params.discountFactor

    def Reset(self):
        self.sess.run(self.init_op) 
    
    def DecisionMakerType(self):
        return "DQN"

    def NumDfltRuns(self):
        return 0

    def TakeDfltValues(self):
        return False

    def GetAllNNVars(self, scope=None):
        if scope == None:
            scope = self.dqnScope

        nnVars = [t for t in tf.trainable_variables() if t.name.find(scope) >= 0]
        nnVars = sorted(nnVars, key=lambda v: v.name)

        npVars = []
        varName = []
        for v in range(len(nnVars)):
            varName.append(nnVars[v].name)
            npVars.append(nnVars[v].eval(session = self.sess))

        return npVars, varName

    def AssignAllNNVars(self, newValues):
        nnVars = [t for t in tf.trainable_variables() if t.name.startswith(self.dqnScope)]
        nnVars = sorted(nnVars, key=lambda v: v.name)

        copy_ops = []
        for v in range(len(nnVars)):
            op = nnVars[v].assign(newValues[v])
            copy_ops.append(op)

        self.sess.run(copy_ops)

    def actionValuesSpecific(self, state, dmId): # dmId = target, curr
        isTarget = dmId == "target"
        return self.ActionsValues(state, isTarget)
    

class DQN_WithTarget(DQN):
    def __init__(self, modelParams, nnName, nnDirectory, agentName = "", isMultiThreaded = False):
        super(DQN_WithTarget, self).__init__(modelParams=modelParams, nnName=nnName, nnDirectory=nnDirectory, isMultiThreaded=isMultiThreaded,
                                                agentName=agentName)
        
        self.numTrials2CmpResults = modelParams.numTrials2CmpResults

        self.targetScope = "TargetNetwork"

        self.lastTrainNumRuns = 0
        
        with tf.variable_scope(self.dqnScope):
            self.valueDqn =tf.get_variable("value_dqn", shape=(), initializer=tf.constant_initializer(-1000.0), dtype=tf.float32)

        with tf.variable_scope(self.targetScope):
            self.numRunsTarget =tf.get_variable("numRuns_target", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32)
            self.valueTarget =tf.get_variable("value_target", shape=(), initializer=tf.constant_initializer(-100.0), dtype=tf.float32)

            # Construct target network
            if modelParams.type == "DQN_Embedding":
                self.targetOutput = self.build_dqn_withEmbedding(modelParams.nn_Func)
            else:
                self.targetOutput = self.build_dqn(modelParams.nn_Func)

        self.rewardHist = []
        if isMultiThreaded:
            self.rewardHistLock = Lock()
        else:
            self.rewardHistLock = EmptyLock()

    def CopyNN(self, scopeTo, scopeFrom):
        fromParams = [t for t in tf.trainable_variables() if t.name.startswith(scopeFrom)]
        fromParams = sorted(fromParams, key=lambda v: v.name)

        toParams = [t for t in tf.trainable_variables() if t.name.startswith(scopeTo)]
        toParams = sorted(toParams, key=lambda v: v.name)

        update_ops = []
        for fromVar, toVar in zip(fromParams, toParams):
            op = toVar.assign(fromVar)
            update_ops.append(op)

        self.sess.run(update_ops)


    def CopyDqn2Target(self, numRuns2Save):
        self.CopyNN(self.targetScope, self.dqnScope)
        
        if numRuns2Save != None:
            assign = self.numRunsTarget.assign(numRuns2Save)
            self.sess.run(assign)

    def CopyTarget2Model(self, numRuns):   
        self.CopyNN(self.dqnScope, self.targetScope)

        assign = self.numRuns.assign(numRuns)
        self.sess.run(assign)
        self.Save()
        
        self.rewardHistLock.acquire()
        self.rewardHist = []
        self.rewardHistLock.release()

    def InitModel(self, sess, resetModel):
        loaded = super(DQN_WithTarget, self).InitModel(sess, resetModel)
        if not loaded:
            self.CopyTarget2Model(0)

    def choose_action(self, state, validActions, targetValues=False):
        if targetValues:   
            vals = self.targetOutput.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess)
            if np.random.uniform() > self.TargetExploreProb():

                maxArgs = list(np.argwhere(vals[0] == np.amax(vals[0][validActions]))[0])
                maxArgsValid = [x for x in maxArgs if x in validActions]
                action = np.random.choice(maxArgsValid)      
            else:
                action = np.random.choice(validActions)
            
            return action, vals
        else:
            return super(DQN_WithTarget, self).choose_action(state, validActions, targetValues)


    def ActionsValues(self, state, validActions, targetValues = False):
        if targetValues:
            allVals = self.targetOutput.eval({self.inputLayer: state.reshape(1,self.num_input)}, session=self.sess)
            return allVals[0]
        else:
            return super(DQN_WithTarget, self).ActionsValues(state, targetValues)


    def TargetExploreProb(self):
        return 0
        
    def NumRunsTarget(self):
        return self.numRunsTarget.eval(session = self.sess)

    def ValueTarget(self):
        return self.valueTarget.eval(session = self.sess)

    def ValueDqn(self):
        return self.valueDqn.eval(session = self.sess)

    def CalcValueDqn(self):
        # calculate results and compare to target
        self.rewardHistLock.acquire()
        rewardHist = self.rewardHist.copy()
        self.rewardHistLock.release()

        if len(rewardHist) >= self.numTrials2CmpResults:
            avgReward = np.average(np.array(rewardHist))
            assign = self.valueDqn.assign(avgReward)
            self.sess.run(assign)

    def learn(self, s, a, r, s_, terminal, numRuns2Save = None): 
        self.CalcValueDqn()
        if self.ValueDqn() > self.ValueTarget():
            self.CopyDqn2Target(self.lastTrainNumRuns)

        self.lastTrainNumRuns = numRuns2Save

        super(DQN_WithTarget, self).learn(s, a, r, s_, terminal, numRuns2Save)
    
    def end_run(self, r, toSave = False):
        super(DQN_WithTarget, self).end_run(r)

        # insert reward to reward history and pop first from histor if necessary
        self.rewardHistLock.acquire()
        self.rewardHist.append(r)
        if len(self.rewardHist) > self.numTrials2CmpResults:
            self.rewardHist.pop(0)
        
        self.rewardHistLock.release()

    def DecisionMakerType(self):
        return "DQN_WithTarget"
        

class DQN_WithTargetAndDefault(DQN_WithTarget):
    def __init__(self, modelParams, nnName, nnDirectory, isMultiThreaded = False, agentName = ""):
        super(DQN_WithTargetAndDefault, self).__init__(modelParams=modelParams, nnName=nnName, nnDirectory=nnDirectory, isMultiThreaded=isMultiThreaded, agentName=agentName)

        self.defaultDecisionMaker = modelParams.defaultDecisionMaker
        self.rewardHistDefault = []
        self.trialsOfDfltRun = modelParams.numTrials2CmpResults
        
        if isMultiThreaded:
            self.rewardHistDfltLock = Lock()
        else:
            self.rewardHistDfltLock = EmptyLock()

        self.defaultScope = "dflt_dm"
        self.initValDflt = 1000.0
        with tf.variable_scope(self.defaultScope):
            self.valueDefaultDm =tf.get_variable("value_dflt", shape=(), initializer=tf.constant_initializer(self.initValDflt), dtype=tf.float32)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

            

    def choose_action(self, state, validActions, targetValues=False):
        if targetValues:
            if self.ValueDefault() > self.ValueTarget():
                return self.defaultDecisionMaker.choose_action(state, validActions, targetValues)
            else:
                return super(DQN_WithTargetAndDefault, self).choose_action(state, validActions, targetValues)
        else:
            if self.ValueDefault() == self.initValDflt:
                return super(DQN_WithTargetAndDefault, self).choose_action(state, validActions)
            else:
                return self.defaultDecisionMaker.choose_action(state, validActions)
    
    def ValueDefault(self):
        return self.valueDefaultDm.eval(session = self.sess)

    def ActionsValues(self, state, validActions, targetValues = False):
        if targetValues:
            if self.ValueDefault() > self.ValueTarget():
                return self.defaultDecisionMaker.ActionsValues(state, targetValues)
            else:
                return super(DQN_WithTargetAndDefault, self).ActionsValues(state, targetValues)
        else:
            if self.ValueDefault() == self.initValDflt:
                return self.defaultDecisionMaker.ActionsValues(state, targetValues)
            else:
                return super(DQN_WithTargetAndDefault, self).ActionsValues(state, targetValues)

    def ExploreProb(self):
        if self.ValueDefault() == self.initValDflt:
            return 0.0
        else:
            return super(DQN_WithTargetAndDefault, self).ExploreProb()

    def end_run(self, r, toSave = False):
        if self.ValueDefault() == self.initValDflt:
            self.rewardHistDfltLock.acquire()
            
            self.rewardHistDefault.append(r)
            if len(self.rewardHistDefault) >= self.trialsOfDfltRun:
                avgReward = np.average(np.array(self.rewardHistDefault))
                assign = self.valueDefaultDm.assign(avgReward)
                self.sess.run(assign)
                self.Save()
            
            self.rewardHistDfltLock.release()
            
            print("\t", threading.current_thread().getName(), " : take default dm value #", len(self.rewardHistDefault))
        else:
            super(DQN_WithTargetAndDefault, self).end_run(r, toSave)

    def DecisionMakerType(self):
        return "DQN_WithTargetAndDefault"
    
    def TakeDfltValues(self):
        return self.ValueDefault() > self.ValueTarget()
    
    def NumDfltRuns(self):
        return len(self.rewardHistDefault)
    
    def DfltValueInitialized(self):
        return self.ValueDefault() != self.initValDflt

    def actionValuesSpecific(self, state, dmId): # dmId = dflt, target, curr
        if dmId == "dflt":
            return self.defaultDecisionMaker.ActionsValues(state)
        else:
            return super(DQN_WithTargetAndDefault, self).actionValuesSpecific(state, dmId)

            
# class CopyDqn:
#     def __init__(self, argListFrom, argListTo):
#         self.sess = tf.Session()
#         argListFrom["params"].tfSession = self.sess