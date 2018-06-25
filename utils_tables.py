import numpy as np
import pandas as pd
import pickle
import os.path
import time
import datetime
import sys

import tensorflow as tf

from utils_dqn import DQN
from utils_dqn import DoubleDQN

from hallucination import HallucinationMngrPSFunc

from multiprocessing import Process, Lock, Value, Array, Manager

# dqn params
class DQN_PARAMS:
    def __init__(self, stateSize, numActions, historyProportion4Learn = 1, nn_Func = None, outputGraph = False, isDoubleDQN = False, discountFactor = 0.95, batchSize = 32, maxReplaySize = 50000, minReplaySize = 1000, copyEvalToTarget = 5, explorationProb = 0.1, descendingExploration = True, exploreChangeRate = 0.0005):
        self.stateSize = stateSize
        self.numActions = numActions
        self.historyProportion4Learn = historyProportion4Learn
        self.nn_Func = nn_Func
        self.discountFactor = discountFactor
        self.batchSize = batchSize
        self.maxReplaySize = maxReplaySize
        self.minReplaySize = minReplaySize
        self.outputGraph = outputGraph
        self.isDoubleDQN = isDoubleDQN
        self.copyEvalToTarget = copyEvalToTarget

        self.explorationProb = explorationProb
        self.descendingExploration = descendingExploration
        self.exploreChangeRate = exploreChangeRate

    def ExplorationProb(self, numRuns):
        if self.descendingExploration:
            return self.explorationProb + (1 - self.explorationProb) * np.exp(-self.exploreChangeRate * numRuns)
        else:
            return self.explorationProb

#qtable params
class QTableParams:
    def __init__(self, stateSize, numActions, historyProportion4Learn = 1, learning_rate=0.01, discountFactor=0.95, explorationProb=0.1, maxReplaySize = 50000, minReplaySize = 200):
        self.stateSize = stateSize
        self.numActions = numActions
        self.learningRate = learning_rate
        self.discountFactor = discountFactor
        self.explorationProb = explorationProb

        self.historyProportion4Learn = historyProportion4Learn

        self.maxReplaySize = maxReplaySize
        self.minReplaySize = minReplaySize

        self.isStateProcessed = True
        
    
    def ExploreProb(self, numRuns):
        return self.explorationProb

    def LearnAtEnd(self):
        return False
    
    def PropogtionUsingTTable(self):
        return False

class QTableParamsWithChangeInExploration:
    def __init__(self, exploreRate = 0.0001, exploreStart = 1, exploreStop = 0.01, learning_rate=0.01, discountFactor=0.95):
        self.learningRate = learning_rate
        self.discountFactor = discountFactor
        self.exploreStart = exploreStart
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate

    def exploreProb(self, numRuns):
        return self.exploreStop + (self.exploreStart - self.exploreStop) * np.exp(-self.exploreRate * numRuns)

    def LearnAtEnd(self):
        return False

    def PropogtionUsingTTable(self):
        return False

class QTablePropogation:
    def __init__(self, learning_rate=0.01, discountFactor=0.95, explorationProb=0.1):
        self.learningRate = learning_rate
        self.discountFactor = discountFactor
        self.explorationProb = explorationProb

    def ExploreProb(self, numRuns):
        return self.explorationProb

    def LearnAtEnd(self):
        return True

    def PropogtionUsingTTable(self):
        return False


class QTablePropogationUsingTTable(QTablePropogation):
    def PropogtionUsingTTable(self):
        return True

class UserPlay:
    def __init__(self, playWithInput = True):
        self.playWithInput = playWithInput
    def choose_action(self, observation):
        if self.playWithInput:
            a = input("insert action: ")
        else:
            a = 0
        return int(a)

    def learn(self, s, a, r, s_, terminal = False):
        if r != 0:
            print(r)
        return None
    def actionValuesVec(self,state):
        return []
    def end_run(self, r):
        return False

TYPES = ["NN" , "Q"]
class LearnWithReplayMngr:
    def __init__(self, modelType, modelParams, terminalStates, nnDirectory = '', qTableName = '', resultFileName = '', historyFileName = '', numTrials2Learn = 100):
        self.numTrials2Learn = numTrials2Learn
        self.trialNum = 0
        self.discountFactor = modelParams.discountFactor
        self.historyProportion4Learn = modelParams.historyProportion4Learn
        self.maxReplaySize = modelParams.maxReplaySize
        self.minReplaySize = modelParams.minReplaySize

        self.terminalStates = terminalStates

        self.transitions = {}
        self.transitions["s"] = []
        self.transitions["a"] = []
        self.transitions["r"] = []
        self.transitions["s_"] = []
        self.transitions["terminal"] = []

        if "new" in sys.argv:
            loadFiles = False
        else:
            loadFiles = True

        self.chooseActionFromNN = modelType == "NN"
        if self.chooseActionFromNN:
            if modelParams.isDoubleDQN:
                self.dqn = DoubleDQN(modelParams, nnDirectory, loadFiles)
            else:
                self.dqn = DQN(modelParams, nnDirectory, loadFiles)
        else:
            self.dqn = None

        if qTableName != '':
            if self.dqn == None:
                qTableParams = modelParams
            else:
                qTableParams = QTableParams(modelParams.stateSize, modelParams.numActions)
            self.qTable = QLearningTable(qTableParams, qTableName, loadFiles)

        else:
            self.qTable = None

        self.tTable = None

        self.histFileName = historyFileName

        if historyFileName != '':
            self.histFileName += '.gz'
            if os.path.isfile(self.histFileName) and loadFiles:
                self.transitions = pd.read_pickle(self.histFileName, compression='gzip')


        if resultFileName != '':
            self.resultFile = ResultFile(resultFileName, numTrials2Learn, loadFiles)
        else:
            self.resultFile = None
    


    def choose_action(self, observation):
        if self.chooseActionFromNN:
            return self.dqn.choose_action(observation)  
        else:
            return self.qTable.choose_action(str(observation))       

    def actionValuesVec(self, state):
        if self.chooseActionFromNN:
            return self.dqn.actionValuesVec(state)
        else:
            return self.qTable.actionValuesVec(str(state))

    def NumRuns(self):
        if self.chooseActionFromNN:
            return self.dqn.NumRuns()
        else:
            return self.qTable.NumRuns()

    def learn(self, s, a, r, s_, terminal = False):
        self.transitions["s"].append(s)
        self.transitions["a"].append(a)
        self.transitions["r"].append(r)
        self.transitions["s_"].append(s_)
        self.transitions["terminal"].append(terminal)

    def ExperienceReplay(self):            
        while len(self.transitions["r"]) > self.maxReplaySize:
            self.transitions["s"].pop(0)
            self.transitions["a"].pop(0)
            self.transitions["r"].pop(0)
            self.transitions["s_"].pop(0)
            self.transitions["terminal"].pop(0)


        idx4Shuffle = np.arange(len(self.transitions["r"]))
        np.random.shuffle(idx4Shuffle)
        size = int(len(idx4Shuffle) * self.historyProportion4Learn)
        chosenIdx = idx4Shuffle[0:size]
        
        s = np.array(self.transitions["s"])[chosenIdx]
        a = np.array(self.transitions["a"])[chosenIdx]
        r = np.array(self.transitions["r"])[chosenIdx]
        s_ = np.array(self.transitions["s_"])[chosenIdx]
        terminal = np.array(self.transitions["terminal"])[chosenIdx]

        if self.histFileName != '':
            pd.to_pickle(self.transitions, self.histFileName, 'gzip') 

        return s, a, r, s_, terminal

    def TerminalState(self, s):
        for v in self.terminalStates.values():
            if np.array_equal(s, v):
                return True
        return False
    def end_run(self, r, score):
        print("for trial#", self.NumRuns(), ": reward =", r, "score =", score)

        learnAndSave = False
        self.trialNum += 1

        if self.trialNum % self.numTrials2Learn == 0:
            learnAndSave = True

        if self.resultFile != None:
            self.resultFile.end_run(r, score, learnAndSave)
        if self.dqn != None:
            self.dqn.end_run()
        if self.qTable != None:
            self.qTable.end_run(r)
        if self.tTable != None:
            self.tTable.end_run(learnAndSave)

        if learnAndSave:
            print("start training with hist size = ", len(self.transitions["r"]), end = ' ')
            s,a,rVec,s_, terminal = self.ExperienceReplay()
            print("after cutting experience training on size =", len(rVec))
            start = datetime.datetime.now() 
            if self.dqn != None:
                if len(a) > self.minReplaySize:
                    self.dqn.learn(s,a,rVec,s_, terminal)
                    self.dqn.save_network()
            if self.qTable != None:
                self.qTable.learnReplay(s,a,rVec,s_)
                self.qTable.SaveTable()
            diff = datetime.datetime.now() - start
            print("duration(ms) for learning and saving:", diff.seconds * 1000 + diff.microseconds / 1000)

        return True 
    def PrintValues(self, s):
        valsDqn = self.dqn.actionValuesVec(s)
        valsQ = self.qTable.actionValuesVec(str(s))
        print("dqn =", list(valsDqn), "\nQ =", valsQ)

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

class TableMngr:
    def __init__(self, modelParams, qTableName, resultFileName = '', transitionTableName = '', onlineHallucination = False, rewardTableName = '', numTrials2SaveTable = 20, numToWriteResult = 100):
        self.qTable = QLearningTable(modelParams, qTableName)

        self.numTrials2Save = numTrials2SaveTable
        self.numTrials = 0

        self.onlineHallucination = onlineHallucination
        if transitionTableName != '':
            self.createTransitionTable = True
            if modelParams.PropogtionUsingTTable():
                self.tTable = BothWaysTransitionTable(modelParams.numActions, transitionTableName)
                self.qTable.InitTTable(self.tTable)
            else:
                self.tTable = TransitionTable(modelParams.numActions, transitionTableName)
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
            self.sharedMemoryPS["num_actions"] = modelParams.numActions
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
            start = datetime.datetime.now()
            timeout = False
            while notFinished and not timeout:
                notFinished = self.sharedMemoryPS["updateTableFlag"]
                # end = datetime.datetime.now()
                # if (end - start).total_seconds() > 100:
                #     timeout = True
                #     print("\n\n\ntimed out", start, end)

            
            self.qTable.ReadTable()

        return saveTable


class SAR:
    def __init__(self,s,a,r):
        self.s = s
        self.a = a
        self.r = r

class QLearningTable:

    def __init__(self, modelParams, qTableName, loadTable = True):
        self.qTableName = qTableName
        self.actions = list(range(modelParams.numActions))  # a list
        self.table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(qTableName + '.gz') and loadTable:
            self.ReadTable()
        
        self.params = modelParams
        
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

    def InitTTable(self, ttable):
        self.ttable = ttable
        self.reverseTable = ttable.reverseKey
        self.normalTable = ttable.normalKey
        self.timeoutPropogation = 10

    def ReadTable(self):
        self.table = pd.read_pickle(self.qTableName + '.gz', compression='gzip')

    def SaveTable(self):
        self.table.to_pickle(self.qTableName + '.gz', 'gzip') 
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
    def actionValuesVec(self, s):
        state_action = self.table.ix[s, :]
        vals = []
        for a in self.actions:
            vals.append(state_action[a])
        return vals
    def NumRuns(self):
        return self.numTotRuns

    def learnReplay(self, statesVec, actionsVec, rewardsVec, nextStateVec):
        for i in range(len(rewardsVec)):
            s = str(statesVec[i])
            s_ = str(nextStateVec[i])
            self.check_state_exist(s)
            self.check_state_exist(s_)
            self.learnIMP(s, actionsVec[i], rewardsVec[i], s_)


    def learnIMP(self, s, a, r, s_ = "terminal"):
        q_predict = self.table.ix[s, a]
        
        if s_ not in self.terminalStates:
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

            self.learnIMP(s, self.history[idx][1], r, s_)

            s_ = s
            rProp *= self.params.discountFactor
    
        self.history = []

    def RewardPropUsingTTable(self, s, a, r):
        def StateInList(state, sarList):
            for sar2cmp in sarList:
                if sar2cmp.s == state:
                    return True
            return False

        startTime = datetime.datetime.now()
        depth = 0
        stateChecked = 0

        openList = [SAR(s,a,r)]
        closedList = []
        currTime = datetime.datetime.now()
        while (currTime - startTime).total_seconds() <= self.timeoutPropogation and len(openList) > 0:
            depth += 1
            for sar in openList:
                closedList.append(sar)
                self.learnIMP(sar.s, sar.a, sar.r)
                if sar.s in self.ttable.table[self.reverseTable]:
                    currTable = self.ttable.table[self.reverseTable][sar.s][0]
                    prevStates = list(currTable.index)
                    for prevS in prevStates:
                        stateChecked += 1
                        if not StateInList(prevS, closedList):
                            state_action = self.table.ix[prevS, :]
                            state_action = state_action.reindex(np.random.permutation(state_action.index))    
                            actionChosen = state_action.idxmax()
                            if currTable.ix[prevS, actionChosen] > 0:
                                openList.append(SAR(prevS, actionChosen, r * self.params.discountFactor))
                
                openList.remove(sar)
            
            currTime = datetime.datetime.now()

        return depth, len(closedList), stateChecked


    def learn(self, s, a, r, s_, sToInitValues, s_ToInitValues):
        self.check_state_exist(s, sToInitValues)
        if s_ not in self.terminalStates:
            self.check_state_exist(s_, s_ToInitValues) 

        if not self.params.LearnAtEnd():
            self.learnIMP(s, a, r, s_)
        else:
            if not self.params.PropogtionUsingTTable():
                self.history.append([s, a, r])
                if s_ in self.terminalStates:
                    self.RewardProp(s_)
            else:
                if s_ not in self.terminalStates:
                    self.learnIMP(s, a, r, s_)
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
            self.table[self.TrialsData][self.NumRunsTotalSlot] = self.numTotRuns
            pd.to_pickle(self.table, self.tableName + '.gz', 'gzip') 

class BothWaysTransitionTable(TransitionTable):
    def __init__(self, numActions, tableName):
        super(BothWaysTransitionTable, self).__init__(numActions, tableName)
        self.normalKey = 0
        self.reverseKey = 1

        if self.normalKey not in self.table:
            self.table[self.normalKey] = {}

        if self.reverseKey not in self.table:
            self.table[self.reverseKey] = {}
            
    def check_item_exist(self, item, tableType):
        if item not in self.table[tableType]:
            # append new state to q table
            self.table[tableType][item] = [None, None]
            self.table[tableType][item][self.tableIdx] = pd.DataFrame(columns=self.actions, dtype=np.float)
            self.table[tableType][item][self.actionSumIdx] = [] #np.zeros(self.actions, dtype = int)
            for a in range(0, len(self.actions)):
                self.table[tableType][item][self.actionSumIdx].append(0)

    def check_state_exist(self, s, s_):
        self.check_item_exist(s, self.normalKey)
        self.check_item_exist(s_, self.reverseKey)
        
        if s_ not in self.table[self.normalKey][s][self.tableIdx].index:
            self.table[self.normalKey][s][self.tableIdx] = self.table[self.normalKey][s][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.table[self.normalKey][s][self.tableIdx].columns, name=s_))   

        if s not in self.table[self.reverseKey][s_][self.tableIdx].index:
            self.table[self.reverseKey][s_][self.tableIdx] = self.table[self.reverseKey][s_][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.table[self.reverseKey][s_][self.tableIdx].columns, name=s))   


    def learn(self, s, a, s_):

        self.check_state_exist(s, s_)  
        # update transition

        self.table[self.reverseKey][s_][self.actionSumIdx][a] += 1
        self.table[self.reverseKey][s_][self.tableIdx].ix[s, a] += 1

        self.table[self.normalKey][s][self.actionSumIdx][a] += 1
        self.table[self.normalKey][s][self.tableIdx].ix[s_, a] += 1

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
    def __init__(self, tableName, numToWrite = 100, loadFile = True):
        self.tableName = tableName
        self.numToWrite = numToWrite

        # keys
        self.inMiddleValKey = 'middleCalculationVal'
        self.inMiddleCountKey = 'middleCalculationCount'
        self.countCompleteKey = 'countComplete'
        self.prevNumToWriteKey = 'prevNumToWrite'

        self.rewardCol = list(range(2))
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + '.gz') and loadFile:
            self.result_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        self.check_state_exist(self.prevNumToWriteKey)
        self.check_state_exist(self.inMiddleValKey)
        self.check_state_exist(self.inMiddleCountKey)
        self.check_state_exist(self.countCompleteKey)

        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])
        if numToWrite != self.result_table.ix[self.prevNumToWriteKey, 0]:
            numToWriteKey = "count_" + str(self.countComplete) + "_numToWrite"
            self.result_table.ix[numToWriteKey, 0] = numToWrite
            self.result_table.ix[numToWriteKey, 1] = 0
            self.result_table.ix[self.prevNumToWriteKey, 0] = numToWrite
        
        self.sumReward = self.result_table.ix[self.inMiddleValKey, 0]
        self.sumScore = self.result_table.ix[self.inMiddleValKey, 1]
        self.numRuns = self.result_table.ix[self.inMiddleCountKey, 0]

        if self.numRuns >= numToWrite:
            self.insertEndRun2Table()

    def check_state_exist(self, state):
        if state not in self.result_table.index:
            # append new state to q table
            self.result_table = self.result_table.append(pd.Series([0] * len(self.rewardCol), index=self.result_table.columns, name=state))

    def insertEndRun2Table(self):
            avgReward = self.sumReward / self.numRuns
            avgScore = self.sumScore / self.numRuns

            countKey = str(self.countComplete)

            self.check_state_exist(countKey)
            self.result_table.ix[countKey, 0] = avgReward
            self.result_table.ix[countKey, 1] = avgScore

            self.countComplete += 1
            self.result_table.ix[self.countCompleteKey, 0] = self.countComplete

            self.sumReward = 0
            self.sumScore = 0
            self.numRuns = 0
            print("avg results for", self.numToWrite, "trials: reward =", avgReward, "score =",  avgScore)

    def end_run(self, r, score, saveTable):
        self.sumReward += r
        self.sumScore += score
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
        
        self.result_table.ix[self.inMiddleValKey, 0] = self.sumReward
        self.result_table.ix[self.inMiddleValKey, 1] = self.sumScore
        self.result_table.ix[self.inMiddleCountKey, 0] = self.numRuns
        
        if saveTable:
            self.result_table.to_pickle(self.tableName + '.gz', 'gzip') 
