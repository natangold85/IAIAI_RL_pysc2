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
from utils_qtable import QLearningTable
from utils_results import ResultFile
from utils_ttable import TransitionTable

from hallucination import HallucinationMngrPSFunc

from utils_qtable import QTableParams

import matplotlib.pyplot as plt

from multiprocessing import Process, Lock, Value, Array, Manager

from utils import ParamsBase

class UserPlay:
    def __init__(self, playWithInput = True, numActions = 1, actionDoNothing = 0):
        self.playWithInput = playWithInput
        self.numActions = numActions
        self.actionDoNothing = actionDoNothing

    def choose_action(self, observation):
        if self.playWithInput:
            a = input("insert action: ")
        else:
            a = self.actionDoNothing
        return int(a)

    def learn(self, s, a, r, s_, terminal = False):
        # if r != 0:
        #     print(r)
        return None
    def actionValuesVec(self,state):
        return np.zeros(self.numActions,dtype = float)
    def end_run(self, r, score = 0 ,steps = 0):
        return False
    def ExploreProb(self):
        return 0

TYPES = ["NN" , "QReplay"]
class LearnWithReplayMngr:
    def __init__(self, modelType, modelParams, dqnName = '', qTableName = '', resultFileName = '', historyFileName = '', directory = '', numTrials2Learn = 100, newRun = False):
        if directory != "":
            fullDirectoryName = "./" + directory +"/"
        else:
            fullDirectoryName = directory
        
        self.numTrials2Learn = numTrials2Learn
        self.trialNum = 0
        self.params = modelParams

        self.transitions = {}
        self.transitions["s"] = []
        self.transitions["a"] = []
        self.transitions["r"] = []
        self.transitions["s_"] = []
        self.transitions["terminal"] = []

        if "new" in sys.argv or newRun:
            loadFiles = False
        else:
            loadFiles = True

        self.chooseActionFromNN = modelType == "NN"
        if self.chooseActionFromNN:
            self.dqn = DQN(modelParams, dqnName, fullDirectoryName, loadFiles)
        else:
            self.dqn = None

        if qTableName != '':
            if self.dqn == None:
                qTableParams = modelParams
            else:
                qTableParams = QTableParams(modelParams.stateSize, modelParams.numActions)
            self.qTable = QLearningTable(qTableParams, fullDirectoryName + qTableName, loadFiles)

        else:
            self.qTable = None

        self.tTable = None

        if historyFileName != '':
            self.histFileName = fullDirectoryName + historyFileName + '.gz'
            if os.path.isfile(self.histFileName) and loadFiles:
                self.transitions = pd.read_pickle(self.histFileName, compression='gzip')
        else:
            self.histFileName = historyFileName



        if resultFileName != '':
            self.resultFile = ResultFile(fullDirectoryName + resultFileName, numTrials2Learn, loadFiles)
        else:
            self.resultFile = None
    

    def ExploreProb(self):
        if self.chooseActionFromNN:
            return self.dqn.ExploreProb()
        else:
            return self.qTable.ExploreProb()

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
        while len(self.transitions["r"]) > self.params.maxReplaySize:
            self.transitions["s"].pop(0)
            self.transitions["a"].pop(0)
            self.transitions["r"].pop(0)
            self.transitions["s_"].pop(0)
            self.transitions["terminal"].pop(0)


        idx4Shuffle = np.arange(len(self.transitions["r"]))
        np.random.shuffle(idx4Shuffle)
        size = int(len(idx4Shuffle) * self.params.historyProportion4Learn)
        chosenIdx = idx4Shuffle[0:size]
        
        s = np.array(self.transitions["s"])[chosenIdx]
        a = np.array(self.transitions["a"])[chosenIdx]
        r = np.array(self.transitions["r"])[chosenIdx]
        s_ = np.array(self.transitions["s_"])[chosenIdx]
        terminal = np.array(self.transitions["terminal"])[chosenIdx]

        if self.histFileName != '':
            pd.to_pickle(self.transitions, self.histFileName, 'gzip') 

        return s, a, r, s_, terminal

    def end_run(self, r, score, steps):
        print("for trial#", int(self.NumRuns()), ": reward =", r, "score =", score, "steps =", steps)

        learnAndSave = False
        self.trialNum += 1

        if self.trialNum % self.numTrials2Learn == 0:
            learnAndSave = True

        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, learnAndSave)
        if self.dqn != None:
            self.dqn.end_run()
        if self.qTable != None:
            self.qTable.end_run(r)
        if self.tTable != None:
            self.tTable.end_run(learnAndSave)

        if self.params.propogateReward:
            self.PropogateReward()

        if learnAndSave:
            print("start training with hist size = ", len(self.transitions["r"]), end = ' ')
            s,a,rVec,s_, terminal = self.ExperienceReplay()
            print("after cutting experience training on size =", len(rVec))

            start = datetime.datetime.now() 
            if self.dqn != None:
                if len(a) > self.params.minReplaySize:
                    self.dqn.learn(s,a,rVec,s_, terminal)
                    self.dqn.save_network()
            if self.qTable != None:
                self.qTable.learnReplay(s,a,rVec,s_, terminal)
                self.qTable.SaveTable()
            diff = datetime.datetime.now() - start
            print("duration(ms) for learning and saving:", diff.seconds * 1000 + diff.microseconds / 1000)

        return learnAndSave 

    def ResetAllData(self):
        self.transitions = {}
        self.transitions["s"] = []
        self.transitions["a"] = []
        self.transitions["r"] = []
        self.transitions["s_"] = []
        self.transitions["terminal"] = []

        if self.dqn != None:
            self.dqn.Reset()
        
        if self.qTable != None:
            self.qTable.Reset()

        if self.resultFile != None:
            self.resultFile.Reset()




    def PropogateReward(self):
        lastIdx = len(self.transitions["r"]) - 1
        prevReward = self.transitions["r"][lastIdx]

        for i in range(lastIdx - 1, -1, -1):
            if self.transitions["terminal"][i]:
                break
            
            prevReward *= self.params.discountFactor
            self.transitions["r"][i] += prevReward
            prevReward = self.transitions["r"][i]

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

    def end_run(self, r, score, steps):
        self.resultFile.end_run(r, score, steps, True)

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

    def end_run(self, r, score, steps):
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
            self.resultFile.end_run(r, score, steps, saveTable)

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
