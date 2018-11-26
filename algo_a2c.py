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
                learning_rateActor=0.00001, learning_rateCritic=0.00001, numTrials2CmpResults=1000, outputGraph=True, accumulateHistory=True, 
                numTrials2Learn=None, numTrials2Save=100):

        super(A2C_PARAMS, self).__init__(stateSize=stateSize, numActions=numActions, discountFactor=discountFactor, numTrials2Learn=numTrials2Learn,
                                        numTrials2Save=numTrials2Save, maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, accumulateHistory=accumulateHistory)
        

        self.learning_rateActor = learning_rateActor
        self.learning_rateCritic = learning_rateCritic
        self.batchSize = batchSize
        self.type = "A2C"
        self.numTrials2CmpResults = numTrials2CmpResults
        self.outputGraph = outputGraph
        self.normalizeState = True


class A2C:
    def __init__(self, modelParams, nnName, nnDirectory, isMultiThreaded = False, agentName = ""):
        self.params = modelParams
        self.summaryDirectoryName = "./logs" + nnDirectory.replace(".", "") + "/"

        self.directoryName = nnDirectory + nnName
        self.agentName = agentName

        with tf.variable_scope("meta_data"):
            self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32)

        with tf.variable_scope("critic"):
            self.critic = AC_Critic(modelParams.stateSize, modelParams.numActions, self.params.learning_rateCritic)
        
        with tf.variable_scope("actor"):
            self.actor = AC_Actor(modelParams.stateSize, modelParams.numActions, self.params.learning_rateActor)

        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summaries = [s for s in summary_ops if self.directoryName in s.name]


    def InitModel(self, session, resetModel=False):
        self.sess = session

        if self.params.outputGraph:
            # $ tensorboard --logdir=logs/directory
            tf.summary.FileWriter(self.summaryDirectoryName, self.sess.graph)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta) and not resetModel:
            self.saver.restore(self.sess, self.directoryName)
            loadedDM = True
        else:
            self.Save()
            loadedDM = False
        
        return loadedDM

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
            print("\n\t", threading.current_thread().getName(), " : ", self.agentName, "->save dqn with", numRuns2Save, "runs to:", self.directoryName)

    def choose_action(self, state, validActions, targetValues=False):
        actionProbs = self.ActionsValues(state, validActions)        
        action = np.random.choice(np.arange(len(actionProbs)), p=actionProbs)

        return action

    def DisperseNonValidValues(self, values, validActions):
        # clean non-valid actions from prob
        validValues = np.zeros(len(values), float)

        #take only valid values
        validValues[validActions] = values[validActions]
        # return values normalize to 1 
        return validValues / validValues.sum()


    def ActionsValues(self, state, validActions, targetVals=False):
        actionProbs = self.actor.ActionsValues(state, self.sess) 
        return self.DisperseNonValidValues(actionProbs, validActions)
    
    def learn(self, s, a, r, s_, terminal, numRuns2Save = None):  
        size = len(a)

        rNextState = self.critic.StatesValue(s_, size, self.sess)
        
        # calculating critic value(s, s_)
        criticTargets = r + self.params.discountFactor * np.invert(terminal) * rNextState
        # estimating advantage function
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
        return "A2C"

    def TakeDfltValues(self):
        return False
    
    def end_run(self, reward):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)

    def Reset(self):
        self.sess.run(self.init_op) 

    def DiscountFactor(self):
        return self.params.discountFactor

class AC_Actor:
    def __init__(self, stateSize, numActions, learningRate):
        # Network Parameters
        self.num_input = stateSize
        self.numActions = numActions        

        self.states = tf.placeholder(tf.float32, shape=[None, self.num_input], name="state")  # (None, 84, 84, 4)
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="targets")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        with tf.variable_scope("NeuralNet"):
            self.actionProb = self.create_actor_nn()

        # We add entropy to the loss to encourage exploration
        with tf.variable_scope("entropy"):
            self.entropy = -tf.reduce_sum(self.actionProb * tf.log(self.actionProb), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

        with tf.variable_scope("picked_action_prob"):
            batch_size = tf.shape(self.actions)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.actionProb)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.actionProb, [-1]), gather_indices)

        self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)

        self.loss = tf.reduce_sum(self.losses, name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(learningRate, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)


        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
        tf.summary.histogram(self.entropy.op.name, self.entropy)


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
        probs = self.actionProb.eval({ self.states: state.reshape(1,self.num_input) }, session=sess)        
        return probs[0]


class AC_Critic:
    def __init__(self, stateSize, numActions, learningRate):
        # Network Parameters
        self.num_input = stateSize
        self.numActions = numActions        
        
        self.states = tf.placeholder(tf.float32, shape=[None, self.num_input], name="state")  # (None, 84, 84, 4)
        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        with tf.variable_scope("NeuralNet"):
            self.output = self.create_critic_nn()

        self.losses = tf.squared_difference(self.output, self.targets)
        self.loss = tf.reduce_sum(self.losses, name="loss")
        
        self.optimizer = tf.train.RMSPropOptimizer(learningRate, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    # Define the neural network
    def create_critic_nn(self, numLayers=2, numNeuronsInLayer=256):
        currInput = self.states
        for i in range(numLayers):
            fc = tf.contrib.layers.fully_connected(currInput, numNeuronsInLayer)
            currInput = fc

        output = tf.contrib.layers.fully_connected(currInput, 1, activation_fn=None)
        output = tf.squeeze(output, squeeze_dims=[1], name="logits")
        return output

    
    def StatesValue(self, states, size, sess):
        return self.output.eval({self.states: states.reshape(size, self.num_input)}, session=sess)

