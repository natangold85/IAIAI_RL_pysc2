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
class A3C_PARAMS(ParamsBase):
    def __init__(self, stateSize, numActions, discountFactor = 0.95, batchSize = 32, maxReplaySize = 500000, minReplaySize = 1000, 
                learning_rate=0.00001, numTrials2CmpResults=1000, outputGraph=False, accumulateHistory=False):

        super(A3C_PARAMS, self).__init__(stateSize=stateSize, numActions=numActions, discountFactor=discountFactor, 
                                            maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, accumulateHistory=accumulateHistory)
        

        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.type = "A3C"
        self.numTrials2CmpResults = numTrials2CmpResults
        self.outputGraph = outputGraph
        self.normalizeRewards = False
        self.numRepeatsTerminalLearning = 0


class A3C:
    def __init__(self, modelParams, nnName, nnDirectory, loadNN, isMultiThreaded = False, agentName = "", createSaver = True):
        self.params = modelParams
        self.directoryName = nnDirectory + nnName
        self.agentName = agentName

        with tf.variable_scope(self.directoryName):
            self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32)

        self.critic = A3C_Critic(modelParams.stateSize, modelParams.numActions, self.directoryName + "_critic")
        self.actor = A3C_Actor(modelParams.stateSize, modelParams.numActions, self.directoryName + "_actor")

        self.sess = tf.Session()    
        if modelParams.outputGraph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter(nnDirectory + "/", self.sess.graph)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

        if createSaver:
            self.saver = tf.train.Saver()
            fnameNNMeta = self.directoryName + ".meta"
            if os.path.isfile(fnameNNMeta) and loadNN:
                self.saver.restore(self.sess, self.directoryName)
            else:
                self.Save()
        else:
            self.saver = None

    def NumRuns(self):
        return self.numRuns.eval(session = self.sess)

    def Save(self, numRuns2Save = None, toPrint = True):
        
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

    def choose_action(self, state, validActions, targetValues=False):
        actionProbs = self.ActionsValues(state, validActions)
        validActionProbs = self.DisperseNonValidValues(actionProbs, validActions)
        
        action = np.random.choice(np.arange(len(validActionProbs)), p=validActionProbs)

        return action

    def DisperseNonValidValues(self, values, validActions):
        # clean non-valid actions from prob
        validValues = np.zeros(len(values), float)

        #take only valid values
        validValues[validActions] = values[validActions]
        # return values normlie to 1 
        return validValues / validValues.sum()


    def ActionsValues(self, state, validActions, targetVals=False):
        return self.actor.ActionsValues(state, self.sess)
    
    def learn(self, s, a, r, s_, terminal, numRuns2Save = None):  
        size = len(a)

        rNextState = self.critic.StatesValue(s_, size, self.sess)
        
        
        criticTargets = r + self.params.discountFactor * np.invert(terminal) * rNextState
        actorTarget = criticTargets - self.critic.StatesValue(s, size, self.sess)

        for i in range(int(size  / self.params.batchSize)):
            chosen = np.arange(i * self.params.batchSize, (i + 1) * self.params.batchSize)
            feedDict = {self.actor.states: s[chosen].reshape(self.params.batchSize, self.params.stateSize), 
                        self.critic.states: s[chosen].reshape(self.params.batchSize, self.params.stateSize),
                        self.actor.actions: a[chosen],
                        self.actor.targets: actorTarget[chosen],
                        self.critic.targets: criticTargets[chosen]}

            self.sess.run([self.actor.loss, self.critic.loss, self.actor.train_op, self.critic.train_op], feedDict)
    
    
    def DecisionMakerType(self):
        return "A3C"
    
    def ExploreProb(self):
        return 0

    def TargetExploreProb(self):
        return 0

    def TakeDfltValues(self):
        return False
    
    def end_run(self, reward, currentNumRuns):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)

    def Reset(self):
        self.sess.run(self.init_op) 

    def DiscountFactor(self):
        return self.params.discountFactor

class A3C_Actor:
    def __init__(self, stateSize, numActions, scope):
        # Network Parameters
        self.num_input = stateSize
        self.numActions = numActions        
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.states = tf.placeholder(tf.float32, shape=[None, self.num_input], name="state")  # (None, 84, 84, 4)
            # The TD target value
            self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
            # Integer id of which action was selected
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            self.output = self.create_actor_nn(self.scope)
            self.actionProb = tf.nn.softmax(self.output) + 1e-8

            # We add entropy to the loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.actionProb * tf.log(self.actionProb), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            batch_size = tf.shape(self.states)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.actionProb)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.actionProb, [-1]), gather_indices)

            self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)

            self.loss = tf.reduce_sum(self.losses, name="loss")
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=tf.contrib.framework.get_global_step())



    # Define the neural network
    def create_actor_nn(self, scope, numLayers=2, numNeuronsInLayer=256):
        with tf.variable_scope(scope):
            currInput = self.states
            for i in range(numLayers):
                fc = tf.contrib.layers.fully_connected(currInput, numNeuronsInLayer)
                currInput = fc

            output = tf.contrib.layers.fully_connected(currInput, self.numActions, activation_fn=None)
            
        return output

    def ActionsValues(self, state, sess):
        probs = self.actionProb.eval({ self.states: state.reshape(1,self.num_input) }, session=sess)
        vals = self.output.eval({ self.states: state.reshape(1,self.num_input) }, session=sess)
        
        return probs[0]


class A3C_Critic:
    def __init__(self, stateSize, numActions, scope):
        # Network Parameters
        self.num_input = stateSize
        self.numActions = numActions        
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.states = tf.placeholder(tf.float32, shape=[None, self.num_input], name="state")  # (None, 84, 84, 4)
            # The TD target value
            self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

            self.output = self.create_critic_nn(self.scope)

            self.losses = tf.squared_difference(self.output, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")
            
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)


    # Define the neural network
    def create_critic_nn(self, scope, numLayers=2, numNeuronsInLayer=256):
        with tf.variable_scope(scope):
            currInput = self.states
            for i in range(numLayers):
                fc = tf.contrib.layers.fully_connected(currInput, numNeuronsInLayer)
                currInput = fc

            output = tf.contrib.layers.fully_connected(currInput, 1, activation_fn=None)
            output = tf.squeeze(output, squeeze_dims=[1], name="logits")
        return output

    
    def StatesValue(self, states, size, sess):
        return self.output.eval({self.states: states.reshape(size, self.num_input)}, session=sess)

