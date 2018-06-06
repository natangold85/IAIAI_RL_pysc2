import numpy as np
import pandas as pd
import pickle
import os.path
import time

import tensorflow as tf

from hallucination import HallucinationMngrPSFunc

from multiprocessing import Process, Lock, Value, Array, Manager


#qtable types
class QTableParamsWOChangeInExploration:
    def __init__(self, learning_rate=0.01, reward_decay=0.9, explorationProb=0.1):
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.explorationProb = explorationProb
    
    def ExploreProb(self, numRuns):
        return self.explorationProb

    def LearnAtEnd(self):
        return False

class QTableParamsWithChangeInExploration:
    def __init__(self, exploreRate = 0.0001, exploreStart = 1, exploreStop = 0.01, learning_rate=0.01, reward_decay=0.9):
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.exploreStart = exploreStart
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate

    def exploreProb(self, numRuns):
        return self.exploreStop + (self.exploreStart - self.exploreStop) * np.exp(-self.exploreRate * numRuns)

    def LearnAtEnd(self):
        return False

class QTablePropogation:
    def __init__(self, learning_rate=0.01, reward_decay=0.95, explorationProb=0.1):
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.explorationProb = explorationProb

    def ExploreProb(self, numRuns):
        return self.explorationProb

    def LearnAtEnd(self):
        return True

class UserPlay:
    def choose_action(self, observation):
        a = input("insert action: ")
        return int(a)
    def learn(self, s, a, r, s_, initValuesState = None):
        return False
    def end_run(self, r):
        return False

class TestTableMngr:        
    def __init__(self, numActions, qTableName, resultFileName, numToWriteResult = 100):
        
        self.qTable = QLearningTable(numActions, qTableName)
        self.resultFile = ResultFile(resultFileName, numToWriteResult)


    def choose_action(self, observation):
        return self.qTable.choose_action(observation)

    def learn(self, s, a, r, s_, sToInitValues = None, s_ToInitValues = None):
        return 

    def end_run(self, r):
        self.resultFile.end_run(r, True)

        return True 

def Foo():
    while True:
        a = 9

class TableMngr:
    def __init__(self, numActions, sizeState, qTableName, resultFileName = '', QTableParams = QTableParamsWOChangeInExploration(), transitionTableName = '', onlineHallucination = False, rewardTableName = '', numTrials2SaveTable = 20, numToWriteResult = 100):
        self.qTable = QLearningTable(numActions, qTableName, QTableParams)

        self.numTrials2Save = numTrials2SaveTable
        self.numTrials = 0

        self.onlineHallucination = onlineHallucination
        if transitionTableName != '':
            self.createTransitionTable = True
            self.tTable = TransitionTable(numActions, transitionTableName)
        else:
            self.createTransitionTable = False

        if rewardTableName != '':
            self.createRewardTable = True
            self.rTable = RewardTable(rewardTableName)
        else:
            self.createRewardTable = False

        if resultFileName != '':
            self.createResultFile = True
            self.resultFile = ResultFile(resultFileName, numToWriteResult)
        else:
            self.createResultFile = False


        if onlineHallucination:
            manager = Manager()
            self.sharedMemoryPS = manager.dict()
            self.sharedMemoryPS["q_table"] = qTableName
            self.sharedMemoryPS["t_table"] = transitionTableName
            self.sharedMemoryPS["num_actions"] = numActions
            self.sharedMemoryPS["updateStateFlag"] = False
            self.sharedMemoryPS["updateTableFlag"] = False
            self.sharedMemoryPS["nextState"] = None
            self.process = Process(target=HallucinationMngrPSFunc, args=(self.sharedMemoryPS,))
            self.process.daemon = True
            self.process.start()

    def choose_action(self, observation):
        return self.qTable.choose_action(observation)

    def learn(self, s, a, r, s_, sToInitValues = None, s_ToInitValues = None):
        self.qTable.learn(s, a, r, s_, sToInitValues, s_ToInitValues)

        if self.createTransitionTable:
            self.tTable.learn(s, a, s_)

        if self.createRewardTable:
            self.rTable.learn(s_, r)
        
        if self.onlineHallucination:
            self.sharedMemoryPS["updateStateFlag"] = True
            self.sharedMemoryPS["nextState"] = s_

    def end_run(self, r):
        saveTable = False
        self.numTrials += 1
        if self.numTrials == self.numTrials2Save:
            saveTable = True
            self.numTrials = 0

        self.qTable.end_run(r, saveTable)

        if self.createTransitionTable:
            self.tTable.end_run(saveTable)

        if self.createRewardTable:
            self.rTable.end_run(saveTable)

        if self.createResultFile:
            self.resultFile.end_run(r, saveTable)

        if self.onlineHallucination and saveTable:
            self.sharedMemoryPS["updateTableFlag"] = True
            notFinished = True
            while notFinished:
                notFinished = self.sharedMemoryPS["updateTableFlag"]
            
            self.qTable.ReadTable()

        return saveTable


class QLearningTable:
    def __init__(self, numActions, qTableName, qTableParams = QTableParamsWOChangeInExploration()):
        self.qTableName = qTableName
        self.actions = list(range(numActions))  # a list
        self.table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(qTableName + '.gz'):
            self.ReadTable()
        
        self.params = qTableParams
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        self.NumRunsExperimentSlot = 1
        
        self.AvgRewardSlot = 2
        self.AvgRewardExperimentSlot = 3

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

    def ReadTable(self):
        self.table = pd.read_pickle(self.qTableName + '.gz', compression='gzip')

    def choose_action(self, observation):
        self.check_state_exist(observation)
        exploreProb = self.params.ExploreProb(self.numTotRuns)

        if np.random.uniform() > exploreProb:
            # choose best action
            state_action = self.table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learnIMP(self, s, a, r, s_):
        q_predict = self.table.ix[s, a]
        
        if s_ not in self.terminalStates:
            q_target = r + self.params.rewardDecay * self.table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        
        # update
        self.table.ix[s, a] += self.params.learningRate * (q_target - q_predict)

    def learn(self, s, a, r, s_, sToInitValues, s_ToInitValues):
        self.check_state_exist(s, sToInitValues)
        if s_ not in self.terminalStates:
            self.check_state_exist(s_, s_ToInitValues) 

        if not self.params.LearnAtEnd():
            self.learnIMP(s, a, r, s_)
        else:
            self.history.append([s, a, r])
            if s_ in self.terminalStates:
                lastIdx = len(self.history) - 1
                rProp = self.history[lastIdx][2]
                self.history[lastIdx][2] = 0
                for idx in range(lastIdx, -1, -1):
                    r = self.history[idx][2] + rProp
                    s = self.history[idx][0]

                    self.learnIMP(s, self.history[idx][1], r, s_)

                    s_ = s
                    rProp *= self.params.rewardDecay
            
                self.history = []


    def end_run(self, r, saveTable):
    
        self.avgTotReward = (self.numTotRuns * self.avgTotReward + r) / (self.numTotRuns + 1)
        self.avgExpReward = (self.numExpRuns * self.avgExpReward + r) / (self.numExpRuns + 1)
        
        self.numTotRuns += 1
        self.numExpRuns += 1

        self.table.ix[self.TrialsData, self.AvgRewardSlot] = self.avgTotReward
        self.table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = self.avgExpReward

        
        self.table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.table.ix[self.TrialsData, self.NumRunsExperimentSlot] = self.numExpRuns

        print("num total runs = ", self.numTotRuns, "avg total = ", self.avgTotReward)
        print("num experiment runs = ", self.numExpRuns, "avg experiment = ", self.avgExpReward)

        if saveTable:
            self.table.to_pickle(self.qTableName + '.gz', 'gzip') 

    def check_state_exist(self, state, stateToInitValues = None):
        if state not in self.table.index:
            # append new state to q table
            self.table = self.table.append(pd.Series([0] * len(self.actions), index=self.table.columns, name=state))
            
            if stateToInitValues in self.table.index:
                self.table.ix[state,:] = self.table.ix[stateToInitValues, :]

class TransitionTable:
    def __init__(self, numActions, tableName):
        self.tableName = tableName
        self.actions = list(range(numActions))  # a list

        self.table = {}
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        
        self.tableIdx = 0
        self.actionSumIdx = 1

        if os.path.isfile(tableName + '.gz'):
            self.table = pd.read_pickle(tableName + '.gz', compression='gzip')
        else:
            self.table[self.TrialsData] = [0]

        self.numTotRuns = self.table[self.TrialsData][self.NumRunsTotalSlot]

    def check_item_exist(self, item):
        if item not in self.table:
            # append new state to q table
            self.table[item] = [None, None]
            self.table[item][self.tableIdx] = pd.DataFrame(columns=self.actions, dtype=np.float)
            self.table[item][self.actionSumIdx] = []
            for a in range(0, len(self.actions)):
                self.table[item][self.actionSumIdx].append(0)

    def check_state_exist(self, s, s_):
        self.check_item_exist(s)
        if s_ not in self.table[s][self.tableIdx].index:
            # append new state to q table
            self.table[s][self.tableIdx] = self.table[s][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.table[s][self.tableIdx].columns, name=s_))   

                

    def learn(self, s, a, s_):
        self.check_state_exist(s, s_)  

        # update transition
        self.table[s][self.tableIdx].ix[s_, a] += 1
        self.table[s][self.actionSumIdx][a] += 1

    def end_run(self, saveTable):
        self.numTotRuns += 1      
        if saveTable:
            print("transition size = ", len(self.table), "num runs =", self.numTotRuns)
            self.table[self.TrialsData][self.NumRunsTotalSlot] = self.numTotRuns
            pd.to_pickle(self.table, self.tableName + '.gz', 'gzip') 

class RewardTable:
    def __init__(self, tableName):

        self.tableName = tableName
        self.rewardIdx = 0
        self.rewardCol = list(range(1))

        self.r_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.r_table = pd.read_pickle(tableName + '.gz', compression='gzip')
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.r_table.ix[self.TrialsData, self.NumRunsTotalSlot]

    def check_state_exist(self, state):
        if state not in self.r_table.index:
            # append new state to q table
            self.r_table = self.r_table.append(pd.Series([0] * len(self.rewardCol), index=self.r_table.columns, name=state))
            return True

        return False

    def learn(self, s_, r):
        if r != 0:
            if self.check_state_exist(s_):
                # insert values
                self.r_table.ix[s_, self.rewardIdx] = r
        
    def end_run(self, saveTable):
   
        self.numTotRuns += 1      
        self.r_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns

        if saveTable:
            self.r_table.to_pickle(self.tableName + '.gz', 'gzip') 

class ResultFile:
    def __init__(self, tableName, numToWrite = 100):
        self.tableName = tableName
        self.numToWrite = numToWrite

        # keys
        self.inMiddleValKey = 'middleCalculationVal'
        self.inMiddleCountKey = 'middleCalculationCount'
        self.countCompleteKey = 'countComplete'
        self.prevNumToWriteKey = 'prevNumToWrite'

        self.rewardCol = list(range(1))
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.result_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        self.check_state_exist(self.prevNumToWriteKey)
        self.check_state_exist(self.inMiddleValKey)
        self.check_state_exist(self.inMiddleCountKey)
        self.check_state_exist(self.countCompleteKey)

        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])
        if numToWrite != self.result_table.ix[self.prevNumToWriteKey, 0]:
            numToWriteKey = "count_" + str(self.countComplete) + "_numToWrite"
            self.result_table.ix[numToWriteKey, 0] = numToWrite
            self.result_table.ix[self.prevNumToWriteKey, 0] = numToWrite
        
        self.sumReward = self.result_table.ix[self.inMiddleValKey, 0]
        self.numRuns = self.result_table.ix[self.inMiddleCountKey, 0]

        if self.numRuns >= numToWrite:
            self.insertEndRun2Table()

    def check_state_exist(self, state):
        if state not in self.result_table.index:
            # append new state to q table
            self.result_table = self.result_table.append(pd.Series([0] * len(self.rewardCol), index=self.result_table.columns, name=state))

    def insertEndRun2Table(self):
            avgReward = self.sumReward / self.numRuns
            countKey = str(self.countComplete)
            self.check_state_exist(countKey)
            self.result_table.ix[countKey, 0] = avgReward

            self.countComplete += 1
            self.result_table.ix[self.countCompleteKey, 0] = self.countComplete

            self.sumReward = 0
            self.numRuns = 0

    def end_run(self, r, saveTable):
        self.sumReward += r
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
        
        self.result_table.ix[self.inMiddleValKey, 0] = self.sumReward
        self.result_table.ix[self.inMiddleCountKey, 0] = self.numRuns
        
        if saveTable:
            self.result_table.to_pickle(self.tableName + '.gz', 'gzip') 
