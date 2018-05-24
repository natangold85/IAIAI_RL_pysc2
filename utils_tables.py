import numpy as np
import pandas as pd
import pickle
import os.path

class QTableParamsWOChangeInExploration:
    def __init__(self, learning_rate=0.01, reward_decay=0.9, explorationProb=0.1):
        self.isExplorationDecay = False
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.explorationProb = explorationProb

class QTableParamsWithChangeInExploration:
    def __init__(self, exploreRate = 0.0001, exploreStart = 1, exploreStop = 0.01, learning_rate=0.01, reward_decay=0.9):
        self.isExplorationDecay = True
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.exploreStart = exploreStart
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate
        
class UserPlay:
    def choose_action(self, observation):
        a = input("insert action: ")
        return int(a)
    def learn(self, s, a, r, s_):
        return False
    def end_run(self, s, a, r, s_):
        return False
        
class TableMngr:
    def __init__(self, numActions, qTableName, QTableParams, transitionTableName = '', resultFileName = '', rewardTableName = '', numTrials2SaveTable = 20, numToWriteResult = 100):
        self.qTable = QLearningTable(numActions, qTableName, QTableParams)

        self.numTrials2Save = numTrials2SaveTable
        self.numTrials = 0

        if transitionTableName != '':
            self.createTransitionTable = True
            #self.tTable = TransitionTable(numActions, transitionTableName)
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

    def choose_action(self, observation):
        return self.qTable.choose_action(observation)

    def learn(self, s, a, r, s_):
        self.qTable.learn(s, a, r, s_)

        if self.createTransitionTable:
            self.tTable.learn(s, a, s_)

        if self.createRewardTable:
            self.rTable.learn(s_, r)

    def end_run(self, s, a, r, s_):
        saveTable = False
        self.numTrials += 1
        if self.numTrials == self.numTrials2Save:
            saveTable = True
            self.numTrials = 0

        self.qTable.learn(s, a, r, 'terminal')
        self.qTable.end_run(r, saveTable)


        if self.createTransitionTable:
            self.tTable.learn(s, a, s_)  
            self.tTable.end_run(saveTable)

        if self.createRewardTable:
            self.rTable.learn(s_, r)
            self.rTable.end_run(saveTable)

        if self.createResultFile:
            self.resultFile.end_run(r, saveTable)

        return saveTable

        

class QLearningTable:
    def __init__(self, numActions, qTableName, qTableParams):
        self.qTableName = qTableName
        self.actions = list(range(numActions))  # a list
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(qTableName + '.gz'):
            self.q_table = pd.read_pickle(qTableName + '.gz', compression='gzip')
        
        self.params = qTableParams
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        self.NumRunsExperimentSlot = 1
        
        self.AvgRewardSlot = 2
        self.AvgRewardExperimentSlot = 3

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot]
        self.avgTotReward = self.q_table.ix[self.TrialsData, self.AvgRewardSlot]
        self.numExpRuns = 0
        self.avgExpReward = 0

        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 0
        self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 0

        # if self.params.propogateBackward:
        #     self.history = []

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if self.params.isExplorationDecay:
            exploreProb = self.params.exploreStop + (self.params.exploreStart - self.params.exploreStop) * np.exp(-self.params.exploreRate * self.numTotRuns)
        else:
            exploreProb = self.params.explorationProb

        if np.random.uniform() > exploreProb:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s)
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            self.check_state_exist(s_)
            q_target = r + self.params.rewardDecay * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        
        # update
        self.q_table.ix[s, a] += self.params.learningRate * (q_target - q_predict)

        # if self.params.propogateBackward:
        #     self.history.append([s, a])

    def end_run(self, r, saveTable):
    
        self.avgTotReward = (self.numTotRuns * self.avgTotReward + r) / (self.numTotRuns + 1)
        self.avgExpReward = (self.numExpRuns * self.avgExpReward + r) / (self.numExpRuns + 1)
        
        self.numTotRuns += 1
        self.numExpRuns += 1

        self.q_table.ix[self.TrialsData, self.AvgRewardSlot] = self.avgTotReward
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = self.avgExpReward

        
        self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot] = self.numExpRuns

        print("num total runs = ", self.numTotRuns, "avg total = ", self.avgTotReward)
        print("num experiment runs = ", self.numExpRuns, "avg experiment = ", self.avgExpReward)

        # if self.params.propogateBackward:
        #     for idx in range(len(self.history) - 1, -1):
        #         self.learn(self.history[idx][0], self.history[idx][1], )


        if saveTable:
            self.q_table.to_pickle(self.qTableName + '.gz', 'gzip') 

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class TransitionTable:
    def __init__(self, numActions, tableName):
        self.tableName = tableName
        self.actions = list(range(numActions))  # a list

        self.transitionDictionary = {} # pd.Panel(major_axis = [], minor_axis = self.actions)
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        
        self.tableIdx = 0
        self.actionSumIdx = 1

        if os.path.isfile(tableName + '.gz'):
            self.transitionDictionary = pd.read_pickle(tableName + '.gz', compression='gzip')
        else:
            self.transitionDictionary[self.TrialsData] = [0]

        self.numTotRuns = self.transitionDictionary[self.TrialsData][self.NumRunsTotalSlot]

    def check_item_exist(self, item):
        if item not in self.transitionDictionary:
            # append new state to q table
            self.transitionDictionary[item] = [None, None]
            self.transitionDictionary[item][self.tableIdx] = pd.DataFrame(columns=self.actions, dtype=np.float)
            self.transitionDictionary[item][self.actionSumIdx] = []
            for a in range(0, len(self.actions)):
                self.transitionDictionary[item][self.actionSumIdx].append(0)

    def check_state_exist(self, s, s_):
        self.check_item_exist(s)
        if s_ not in self.transitionDictionary[s][self.tableIdx].index:
            # append new state to q table
            self.transitionDictionary[s][self.tableIdx] = self.transitionDictionary[s][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.transitionDictionary[s][self.tableIdx].columns, name=s_))   

                

    def learn(self, s, a, s_):
        self.check_state_exist(s, s_)  

        # update transition
        self.transitionDictionary[s][self.tableIdx].ix[s_, a] += 1
        self.transitionDictionary[s][self.actionSumIdx][a] += 1

    def end_run(self, saveTable):
        self.numTotRuns += 1      
        if saveTable:
            print("transition size = ", len(self.transitionDictionary), "num runs =", self.numTotRuns)
            self.transitionDictionary[self.TrialsData][self.NumRunsTotalSlot] = self.numTotRuns
            pd.to_pickle(self.transitionDictionary, self.tableName + '.gz', 'gzip') 

class TransitionTable2D:
    def __init__(self, numActions, tableName):
        self.tableName = tableName
        self.actions = list(range(numActions))  # a list
        self.t_table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.t_table = pd.read_pickle(tableName + '.gz', compression='gzip')
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.t_table.ix[self.TrialsData, self.NumRunsTotalSlot]

    def check_state_exist(self, state):
        if state not in self.t_table.index:
            # append new state to q table
            self.t_table = self.t_table.append(pd.Series([0] * len(self.actions), index=self.t_table.columns, name=state))

    def learn(self, s, a, s_):
        state = s + "__" + s_
        self.check_state_exist(state)             
        # update transition
        self.t_table.ix[state, a] += 1

    def end_run(self, saveTable):
   
        self.numTotRuns += 1      
        self.t_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns

        if saveTable:
            self.t_table.to_pickle(self.tableName + '.gz', 'gzip') 

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