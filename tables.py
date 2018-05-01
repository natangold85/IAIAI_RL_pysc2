import numpy as np
import pandas as pd
import os.path

class QTableParamsWOChangeInExploration:
    def __init__(self, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.isExplorationDecay = False
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.explorationProb = e_greedy

class QTableParamsWithChangeInExploration:
    def __init__(self, exploreRate = 0.0001, exploreStart = 1, exploreStop = 0.01, learning_rate=0.01, reward_decay=0.9):
        self.isExplorationDecay = True
        self.learningRate = learning_rate
        self.rewardDecay = reward_decay
        self.exploreStart = exploreStart
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate


class TableMngr:
    def __init__(self, numActions, qTableName, QTableParams, transitionTableName = '', rewardTableName = '', resultFileName = '', numToWriteResult = 100):
        self.qTable = QLearningTable(numActions, qTableName, QTableParams)

        if transitionTableName != '':
            self.createTransitionTable = True
            self.tTable = TransitionTable(numActions, transitionTableName)
        else:
            self.createTransitionTable = False

        if rewardTableName != '':
            self.createRewardTable = True
            self.rTable = RewardTable(numActions, rewardTableName)
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
            self.rTable.learn(s, a, r)

    def end_run(self, s, a, r):
        self.qTable.learn(s, a, r, 'terminal')
        self.qTable.end_run(r)

        if self.createTransitionTable:
            if r > 0:
                s_ = 'win'
            elif r < 0:
                s_ = 'loss'
            else:
                s_ = 'tie'
            self.tTable.learn(s, a, s_)  
            self.tTable.end_run()

        if self.createRewardTable:
            self.rTable.learn(s, a, r)
            self.rTable.end_run()

        if self.createResultFile:
            self.resultFile.end_run(r)

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

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if self.params.isExplorationDecay:
            exploreProb = self.params.exploreStop + (self.params.exploreStart - self.params.exploreStop) * np.exp(-self.params.exploreRate * self.numTotRuns)
        else:
            exploreProb = self.params.explorationProb

        if np.random.uniform() < exploreProb:
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

    def end_run(self, r):
    
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

        self.q_table.to_pickle(self.qTableName + '.gz', 'gzip') 

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class TransitionTable:
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

    def end_run(self):
   
        self.numTotRuns += 1      
        self.t_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns

        self.t_table.to_pickle(self.tableName + '.gz', 'gzip') 

class RewardTable:
    def __init__(self, tableName, numActions, learning_rate=0.01):
        self.tableName = tableName
        self.lr = learning_rate

        self.actions =  list(range(numActions))

        self.r_table = pd.DataFrame(columns=self.actions, dtype=np.float)
        if os.path.isfile(tableName + '.gz'):
            self.r_table = pd.read_pickle(tableName + '.gz', compression='gzip')
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.r_table.ix[self.TrialsData, self.NumRunsTotalSlot]

    def check_state_exist(self, state):
        if state not in self.r_table.index:
            # append new state to q table
            self.r_table = self.r_table.append(pd.Series([0] * len(self.actions), index=self.r_table.columns, name=state))

    def learn(self, s, a, r):
        self.check_state_exist(s)             
        # update reward and count

        predictedReward = self.r_table.ix[s, a] 
        # update
        self.r_table.ix[s, a] += self.lr * (r - predictedReward)
        
    def end_run(self):
   
        self.numTotRuns += 1      
        self.r_table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.r_table.to_pickle(self.tableName + '.gz', 'gzip') 

class ResultFile:
    def __init__(self, tableName, numToWrite = 100):
        self.tableName = tableName
        self.numToWrite = numToWrite

        self.file = open(tableName + '.txt', 'w')
        
        self.sumReward = 0
        self.numRuns = 0
    def end_run(self, r):
        self.sumReward += r
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            avgReward = self.sumReward / self.numToWrite
            self.file.write(avgReward, "for", self.numToWrite, "runs")
            self.sumReward = 0
            self.numRuns = 0