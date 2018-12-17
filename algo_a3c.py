import numpy as np
import pandas as pd
import random
import pickle
import os.path
import threading

import tensorflow as tf
import os

import scipy

from utils import ParamsBase

from multiprocessing import Lock


# dqn params
class A3C_PARAMS(ParamsBase):
    def __init__(self, stateSize, numActions, discountFactor = 0.95, batchSize = 32, learning_rate=1e-5, 
                numTrials2CmpResults=1000, outputGraph=True, accumulateHistory=False, numTrials2Learn=1, numTrials2Save=100):

        super(A3C_PARAMS, self).__init__(stateSize=stateSize, numActions=numActions, discountFactor=discountFactor, accumulateHistory=accumulateHistory, numTrials2Learn=numTrials2Learn)
        

        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.type = "A3C"
        self.numTrials2CmpResults = numTrials2CmpResults
        self.outputGraph = outputGraph
        self.normalizeState = True
        self.numTrials2Save = numTrials2Save

class A3C:
    def __init__(self, modelParams, nnName, nnDirectory, isMultiThreaded = False, agentName = ""):
        self.params = modelParams
        self.summaryDirectoryName = "./logs" + nnDirectory.replace(".", "") + "/"

        self.a3cDirectoryName = nnDirectory + nnName
        self.agentName = agentName

        self.workers = {}

        self.updateGlobalLock = Lock()

        self.trainer = tf.train.AdamOptimizer(learning_rate=modelParams.learning_rate)

        with tf.variable_scope("global_network"):
            self.globalNetwork = A3C_Pair(modelParams, trainer=self.trainer, currScope="global_network")

        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summaries = [s for s in summary_ops if self.a3cDirectoryName in s.name]

    def AddWorker(self, workerName):
        with tf.variable_scope(workerName):
            worker = A3C_Pair(self.params, trainer=self.trainer, currScope=workerName, globalScope="global_network")
        
        self.workers[workerName] = worker
        if len(self.workers) == 1:
            self.worker4Summary = workerName
            self.summary_writer = tf.summary.FileWriter(workerName)

        return worker

    def InitModel(self, session, resetModel=False):
        self.sess = session

        if self.params.outputGraph:
            # $ tensorboard --logdir=logs/directory
            tf.summary.FileWriter(self.summaryDirectoryName, self.sess.graph)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op) 

        # copy params from global network to workers
        for name in self.workers.keys():
            self.CopyParams2AC("global_network", name)

        self.saver = tf.train.Saver()
        fnameNNMeta = self.a3cDirectoryName + ".meta"
        if os.path.isfile(fnameNNMeta) and not resetModel:
            self.saver.restore(self.sess, self.a3cDirectoryName)
            loadedDM = True
        else:
            self.Save()
            loadedDM = False
        
        return loadedDM

    def NumRunsAgent(self, agentName):
        return self.workers[agentName].NumRuns(self.sess)

    def NumRuns(self):
        return self.globalNetwork.NumRuns(self.sess)

    def Save(self, toPrint=True):
        self.saver.save(self.sess, self.a3cDirectoryName)

        if toPrint:
            globalNumRuns = self.globalNetwork.NumRuns(self.sess)
            print("\n\t", threading.current_thread().getName(), " : ", self.agentName, "->save dqn with", globalNumRuns, "runs to:", self.a3cDirectoryName)

    def choose_action(self, state, validActions, targetValues=False):
        actionProbs = self.ActionsValues(state, validActions, targetValues)
        action = np.random.choice(np.arange(len(actionProbs)), p=actionProbs)
        return action, actionProbs

    def DisperseNonValidValues(self, values, validActions):
        # clean non-valid actions from prob
        validValues = np.zeros(len(values), float)

        #take only valid values
        validValues[validActions] = values[validActions]
        # return values normlie to 1 
        return validValues / validValues.sum()


    def ActionsValues(self, state, validActions, targetVals=False):
        if targetVals:
            actionProbs = self.globalNetwork.ActionsValues(state, self.sess)
        else:
            workerName = threading.current_thread().getName()
            actionProbs = self.workers[workerName].ActionsValues(state, self.sess)

        return self.DisperseNonValidValues(actionProbs, validActions)

    # def DiscountReward(self, r, discountFactor):
    #     return scipy.signal.lfilter([1], [1, -discountFactor], r[::-1], axis=0)[::-1]

    def learn(self, s, a, r, s_, terminal, insert2Graph=False): 
        workerName = threading.current_thread().getName()
        acPair = self.workers[workerName]

        size = len(a)
        
        # estimating Q(s, a)
        rNextState = acPair.StatesValue(s_, size, self.sess)
        rNextStateGlobal = self.globalNetwork.StatesValue(s_, size, self.sess)

        qVal = r + self.params.discountFactor * np.invert(terminal) * rNextState        
        # estimating advantage function
        advantage = qVal - acPair.StatesValue(s, size, self.sess)

        # # insert reward to filter
        # discountedQVal = self.DiscountReward(qVal, self.params.discountFactor)
        # discountedAdvantage = self.DiscountReward(advantage, self.params.discountFactor)

        feedDict = {acPair.states: s.reshape(size, self.params.stateSize), acPair.actions: a, acPair.target_v: qVal, acPair.advantages: advantage}
        criticLossBefore, actorLossBefore, varNorms, gradNorms, _ = self.sess.run([acPair.critic_loss, acPair.actor_loss, acPair.var_norms, acPair.grad_norms ,acPair.apply_grads], feedDict)
        
        self.CopyParams2AC("global_network", workerName)

        # if insert2Graph and workerName == self.worker4Summary:
        #     summary = tf.Summary()
        #     summary.value.add(tag='Losses/Critic Loss', simple_value=float(criticLoss / size))
        #     summary.value.add(tag='Losses/Actor Loss', simple_value=float(actorLoss / size))
        #     summary.value.add(tag='Losses/Grad Norm', simple_value=float(gradNorms))
        #     summary.value.add(tag='Losses/Var Norm', simple_value=float(varNorms))
            
        #     self.summary_writer.add_summary(summary)
        #     self.summary_writer.flush()

        return criticLossBefore, actorLossBefore, varNorms, gradNorms

    def CopyParams2AC(self, srcScope, targetScope):
        srcParams = [t for t in tf.trainable_variables() if t.name.find(srcScope) >= 0]
        srcParams = sorted(srcParams, key=lambda v: v.name)

        targetParams = [t for t in tf.trainable_variables() if t.name.find(targetScope) >= 0]
        targetParams = sorted(targetParams, key=lambda v: v.name)

        update_ops = []
        for srcVar, targetVar in zip(srcParams, targetParams):
            op = targetVar.assign(srcVar)
            update_ops.append(op)

        self.sess.run(update_ops)        

    def DecisionMakerType(self):
        return "A3C"
    
    def TakeDfltValues(self):
        return False
    
    def end_run(self, reward):
        workerName = threading.current_thread().getName()
        assign = self.workers[workerName].numRuns.assign_add(1)
        self.sess.run(assign)

    def Reset(self):
        self.sess.run(self.init_op) 

    def DiscountFactor(self):
        return self.params.discountFactor


class A3C_Pair:
    def __init__(self, params, trainer, currScope, globalScope=None):
        # Network Parameters
        self.stateSize = params.stateSize
        self.numActions = params.numActions        

        self.states = tf.placeholder(shape=[None, self.stateSize],dtype=tf.float32, name="state")

        with tf.variable_scope("meta_data"):
            self.numRuns = tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32, trainable=False)
            if globalScope != None:
                self.numRunsLastCopied = tf.get_variable("numRunsLastCopied", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32, trainable=False)

        with tf.variable_scope("NeuralNetActor"):
            self.actor = self.create_actor_nn()

        with tf.variable_scope("NeuralNetCritic"):
            self.critic = self.create_critic_nn()

        if globalScope != None:

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
            self.actions_onehot = tf.one_hot(self.actions, self.numActions, dtype=tf.float32, name="actions_onehot")
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32, name="target_value")
            self.advantages = tf.placeholder(shape=[None],dtype=tf.float32, name="advantages")

            self.responsible_outputs = tf.reduce_sum(self.actor * self.actions_onehot, [1])

            with tf.variable_scope("LossFunction"):
                #Loss functions
                self.actor_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.critic,[-1])))
                self.entropy = - tf.reduce_sum(self.actor * tf.log(self.actor))
                self.critic_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.actor_loss + self.critic_loss - self.entropy * 0.01

            with tf.variable_scope("ComputeGradients"):
                #Get gradients from local network using local losses
                local_vars = [t for t in tf.trainable_variables() if t.name.find(currScope) > 0]
                global_vars = [t for t in tf.trainable_variables() if t.name.find(globalScope) > 0]

                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
       
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                #Apply local gradients to global network
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


    # Define the neural network
    def create_critic_nn(self, numLayers=2, numNeuronsInLayer=256):
        currInput = self.states
        for i in range(numLayers):
            fc = tf.contrib.layers.fully_connected(currInput, numNeuronsInLayer)
            currInput = fc

        output = tf.contrib.layers.fully_connected(currInput, 1, activation_fn=tf.nn.sigmoid) * 2 - 1
        output = tf.squeeze(output, squeeze_dims=[1], name="Critic_Output")
        return output

    # Define the neural network
    def create_actor_nn(self, numLayers=2, numNeuronsInLayer=256):
        
        currInput = self.states
        for i in range(numLayers):
            fc = tf.contrib.layers.fully_connected(currInput, numNeuronsInLayer)
            currInput = fc

        output = tf.contrib.layers.fully_connected(currInput, self.numActions, activation_fn=None)
        softmax = tf.nn.softmax(output) + 1e-8

        return softmax

    def ActionsValues(self, state, sess):
        probs = self.actor.eval({ self.states: state.reshape(1,self.stateSize) }, session=sess)        
        return probs[0]

    def StatesValue(self, state, size, sess):
        val = self.critic.eval({ self.states: state.reshape(size, self.stateSize) }, session=sess) 
        return val 

    def NumRuns(self, sess):
        return self.numRuns.eval(session=sess)
    
    def NumRunsLastCopied(self, sess):
        return self.numRunsLastCopied.eval(session=sess)