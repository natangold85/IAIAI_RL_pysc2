import math
import numpy as np
import pandas as pd
import pickle
import os.path
import time
import datetime
import sys

import tensorflow as tf

from utils_dqn import DQN
from hallucination import HallucinationMngrPSFunc

import matplotlib.pyplot as plt

from multiprocessing import Process, Lock, Value, Array, Manager

from utils import ParamsBase

#qtable params
class QTableParams(ParamsBase):
    def __init__(self, stateSize, numActions, historyProportion4Learn = 1, propogateReward = False, learning_rate=0.01, discountFactor=0.95, explorationProb=0.1, maxReplaySize = 50000, minReplaySize = 1000, states2Monitor = []):
        super(QTableParams, self).__init__(stateSize, numActions, historyProportion4Learn, propogateReward, discountFactor, maxReplaySize, minReplaySize, states2Monitor)    
        self.learningRate = learning_rate
        self.explorationProb = explorationProb        
    
    def ExploreProb(self, numRuns):
        return self.explorationProb

    def LearnAtEnd(self):
        return False
    
    def PropogtionUsingTTable(self):
        return False

class QTableParamsExplorationDecay(ParamsBase):
    def __init__(self, stateSize, numActions, historyProportion4Learn = 1, propogateReward = False, learning_rate=0.01, discountFactor=0.95, exploreRate = 0.0005, exploreStop = 0.1, maxReplaySize = 50000, minReplaySize = 1000, states2Monitor = []):
        super(QTableParamsExplorationDecay, self).__init__(stateSize, numActions, historyProportion4Learn, propogateReward, discountFactor, maxReplaySize, minReplaySize, states2Monitor) 

        self.learningRate = learning_rate        
        self.exploreStart = 1
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate

    def ExploreProb(self, numRuns):
        return self.exploreStop + (self.exploreStart - self.exploreStop) * np.exp(-self.exploreRate * numRuns)

    def LearnAtEnd(self):
        return False

    def PropogtionUsingTTable(self):
        return False

class SAR:
    def __init__(self,s,a,r):
        self.s = s
        self.a = a
        self.r = r

class QLearningTable:

    def __init__(self, modelParams, qTableName, loadTable = True):
        self.qTableName = qTableName
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        self.NumRunsExperimentSlot = 1
        self.AvgRewardSlot = 2
        self.AvgRewardExperimentSlot = 3

        slotsInTable = max(4, modelParams.numActions)
        self.actions = list(range(modelParams.numActions))
        self.slots = list(range(slotsInTable))  # a list
        self.table = pd.DataFrame(columns=self.slots, dtype=np.float)
        if os.path.isfile(qTableName + '.gz') and loadTable:
            self.ReadTable()
        
        self.params = modelParams
        

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.table.ix[self.TrialsData, self.NumRunsTotalSlot]
        self.avgTotReward = self.table.ix[self.TrialsData, self.AvgRewardSlot]
        self.numExpRuns = 0
        self.avgExpReward = 0

        self.table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 0
        self.table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 0

        self.terminalStates = ['terminal', 'win', 'loss', 'tie']

        if self.params.LearnAtEnd():
            self.history = []

    def InitTTable(self, ttable):
        self.ttable = ttable
        self.reverseTable = ttable.reverseKey
        self.normalTable = ttable.normalKey
        self.timeoutPropogation = 10

    def ReadTable(self):
        self.table = pd.read_pickle(self.qTableName + '.gz', compression='gzip')

    def SaveTable(self):
        self.table.to_pickle(self.qTableName + '.gz', 'gzip') 
    
    def choose_absolute_action(self, observation):
        self.check_state_exist(observation)
        state_action = self.table.ix[observation, self.actions]
        
        state_actionReindex = state_action.reindex(np.random.permutation(state_action.index))
        action = state_actionReindex.idxmax()

        return action, state_action[action]

    def ExploreProb(self):
        return self.params.ExploreProb(self.numTotRuns)

        #self.check_state_exist(observation)
    def choose_action(self, observation):

        exploreProb = self.params.ExploreProb(self.numTotRuns)

        if np.random.uniform() > exploreProb:
            # choose best action
            state_action = self.table.ix[observation, self.actions]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def actionValuesVec(self, s):
        self.check_state_exist(s)
        state_action = self.table.ix[s, :]
        vals = np.zeros(len(self.actions), dtype=float)
        for a in range(len(self.actions)):
            vals[a] = state_action[self.actions[a]]

        return vals
    def NumRuns(self):
        return self.numTotRuns

    def learnReplay(self, statesVec, actionsVec, rewardsVec, nextStateVec, terminal):
        for i in range(len(rewardsVec)):
            s = str(statesVec[i])
            s_ = str(nextStateVec[i])
            self.check_state_exist(s)
            self.check_state_exist(s_)
            self.learnIMP(s, actionsVec[i], rewardsVec[i], s_, terminal[i])

        for i in range(len(self.params.states2Monitor)):
            state = str(self.params.states2Monitor[i][0])
            actions2Print = self.params.states2Monitor[i][1]
            vals = self.actionValuesVec(state)
            for a in actions2Print:
                print(vals[a], end = ", ")           
            print("\n")

    def learnIMP(self, s, a, r, s_, terminal):
        q_predict = self.table.ix[s, a]
        
        if not terminal:
            q_target = r + self.params.discountFactor * self.table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        
        # update
        self.table.ix[s, a] += self.params.learningRate * (q_target - q_predict)

    def RewardProp(self, s_):
        lastIdx = len(self.history) - 1
        rProp = self.history[lastIdx][2]
        self.history[lastIdx][2] = 0
        for idx in range(lastIdx, -1, -1):
            r = self.history[idx][2] + rProp
            s = self.history[idx][0]
            terminal = s_ in self.terminalStates
            self.learnIMP(s, self.history[idx][1], r, s_, terminal)

            s_ = s
            rProp *= self.params.discountFactor
    
        self.history = []

    # def RewardPropUsingTTable(self, s, a, r):
    #     def StateInList(state, sarList):
    #         for sar2cmp in sarList:
    #             if sar2cmp.s == state:
    #                 return True
    #         return False

    #     startTime = datetime.datetime.now()
    #     depth = 0
    #     stateChecked = 0

    #     openList = [SAR(s,a,r)]
    #     closedList = []
    #     currTime = datetime.datetime.now()
    #     while (currTime - startTime).total_seconds() <= self.timeoutPropogation and len(openList) > 0:
    #         depth += 1
    #         for sar in openList:
    #             closedList.append(sar)
    #             self.learnIMP(sar.s, sar.a, sar.r)
    #             if sar.s in self.ttable.table[self.reverseTable]:
    #                 currTable = self.ttable.table[self.reverseTable][sar.s][0]
    #                 prevStates = list(currTable.index)
    #                 for prevS in prevStates:
    #                     stateChecked += 1
    #                     if not StateInList(prevS, closedList):
    #                         state_action = self.table.ix[prevS, :]
    #                         state_action = state_action.reindex(np.random.permutation(state_action.index))    
    #                         actionChosen = state_action.idxmax()
    #                         if currTable.ix[prevS, actionChosen] > 0:
    #                             openList.append(SAR(prevS, actionChosen, r * self.params.discountFactor))
                
    #             openList.remove(sar)
            
    #         currTime = datetime.datetime.now()

    #     return depth, len(closedList), stateChecked


    def learn(self, s, a, r, s_, sToInitValues, s_ToInitValues):
        self.check_state_exist(s, sToInitValues)
        if s_ not in self.terminalStates:
            self.check_state_exist(s_, s_ToInitValues) 

        if not self.params.LearnAtEnd():
            terminal = s_ in self.terminalStates
            self.learnIMP(s, a, r, s_, terminal)
        else:
            if not self.params.PropogtionUsingTTable():
                self.history.append([s, a, r])
                if s_ in self.terminalStates:
                    self.RewardProp(s_)
            else:
                if s_ not in self.terminalStates:
                    self.learnIMP(s, a, r, s_, False)
                else:
                    start = datetime.datetime.now()
                    depth,size, checked = self.RewardPropUsingTTable(s, a, r)
                    diff = datetime.datetime.now() - start
                    print("propogation time", diff.seconds * 1000 + diff.microseconds / 1000, "size =", size, "depth =", depth, "checked =", checked)


    def end_run(self, r, saveTable = False):
        self.avgTotReward = (self.numTotRuns * self.avgTotReward + r) / (self.numTotRuns + 1)
        self.avgExpReward = (self.numExpRuns * self.avgExpReward + r) / (self.numExpRuns + 1)
        
        self.numTotRuns += 1
        self.numExpRuns += 1

        self.table.ix[self.TrialsData, self.AvgRewardSlot] = self.avgTotReward
        self.table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = self.avgExpReward

        
        self.table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.table.ix[self.TrialsData, self.NumRunsExperimentSlot] = self.numExpRuns

        # print("num total runs = ", self.numTotRuns, "avg total = ", self.avgTotReward)
        # print("num experiment runs = ", self.numExpRuns, "avg experiment = ", self.avgExpReward)

        if saveTable:
            self.table.to_pickle(self.qTableName + '.gz', 'gzip') 

    def check_state_exist(self, state, stateToInitValues = None):
        if state not in self.table.index:
            # append new state to q table
            self.table = self.table.append(pd.Series([0] * len(self.slots), index=self.table.columns, name=state))
            
            if stateToInitValues in self.table.index:
                self.table.ix[state,:] = self.table.ix[stateToInitValues, :]
            return True

        return False