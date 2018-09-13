import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys
import threading

from multiprocessing import Lock

from utils import EmptyLock

#decision makers
from utils_dqn import DQN
from utils_dqn import DQN_WithTarget

from utils_qtable import QLearningTable

from utils_history import HistoryMngr

# model builders:
from utils_ttable import TransitionTable

# results handlers
from utils_results import ResultFile

class BaseDecisionMaker:
    def __init__(self):
        self.trainFlag = False
        self.switchFlag = False

        self.subAgentsDecisionMakers = {}
        self.switchCount = {}
        self.resultFile = None

    def SetSubAgentDecisionMaker(self, key, decisionMaker):
        self.subAgentsDecisionMakers[key] = decisionMaker
        self.switchCount[key] = 0

    def GetSubAgentDecisionMaker(self, key):
        if key in self.subAgentsDecisionMakers.keys():
            return self.subAgentsDecisionMakers[key]
        else:
            return None

    def TrainAll(self):
        if self.trainFlag:
            self.Train()
            self.trainFlag = False
        
        for subDM in self.subAgentsDecisionMakers.values():
            if subDM != None:
                subDM.TrainAll()

    def AddSwitch(self, idx, numSwitch, name):
        if self.resultFile != None and idx in self.switchCount:
            if self.switchCount[idx] <= numSwitch:
                self.switchCount[idx] = numSwitch + 1
                slotName = name + "_" + str(numSwitch)
                self.resultFile.AddSwitchSlot(slotName)

    def Train(self):
        pass      
    def NumRuns(self):
        pass
    def choose_action(self, observation):
        pass
    def learn(self, s, a, r, s_, terminal = False):
        pass
    def ActionValuesVec(self,state, targetValues = False):
        pass
    def end_run(self, r, score = 0 ,steps = 0):
        pass
    def ExploreProb(self):
        return 0
    def TargetExploreProb(self):
        return 0

    def TrimHistory(self):
        pass
    
    def AddHistory(self):
        return None
        
class BaseNaiveDecisionMaker(BaseDecisionMaker):
    def __init__(self, numTrials2Save, resultFName = None, directory = None):
        super(BaseNaiveDecisionMaker, self).__init__()
        self.resultFName = resultFName
        self.trialNum = 0
        self.numTrials2Save = numTrials2Save

        if resultFName != None:
            self.lock = Lock()
            if directory != None:
                fullDirectoryName = "./" + directory +"/"
                if not os.path.isdir(fullDirectoryName):
                    os.makedirs(fullDirectoryName)
            else:
                fullDirectoryName = "./"

            if "new" in sys.argv:
                loadFiles = False
            else:
                loadFiles = True

            self.resultFile = ResultFile(fullDirectoryName + resultFName, numTrials2Save, loadFiles)

    def end_run(self, r, score, steps):
        saveFile = False
        self.trialNum += 1
        
        if self.resultFName != None:
            self.lock.acquire()
            if self.trialNum % self.numTrials2Save == 0:
                saveFile = True

            self.resultFile.end_run(r, score, steps, saveFile)
            self.lock.release()
       
        return saveFile 


class UserPlay(BaseDecisionMaker):
    def __init__(self, playWithInput = True, numActions = 1, actionDoNothing = 0):
        super(UserPlay, self).__init__()
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
        return None

    def ActionValuesVec(self,state, targetValues = False):
        return np.zeros(self.numActions,dtype = float)

    def end_run(self, r, score = 0 ,steps = 0):
        return False

    def ExploreProb(self):
        return 0


class LearnWithReplayMngr(BaseDecisionMaker):
    def __init__(self, modelType, modelParams, decisionMakerName = '', resultFileName = '', historyFileName = '', directory = '', numTrials2Learn = 100, isMultiThreaded = False):
        super(LearnWithReplayMngr, self).__init__()

        # params
        self.params = modelParams
        self.numTrials2Learn = numTrials2Learn

        # trial to DQN to learn
        self.trial2LearnDQN = -1
        
        if directory != "":
            fullDirectoryName = "./" + directory +"/"
            if not os.path.isdir(fullDirectoryName):
                os.makedirs(fullDirectoryName)
        else:
            fullDirectoryName = "./"
        
    
        if "new" in sys.argv:
            loadFiles = False
        else:
            loadFiles = True

        decisionClass = eval(modelType)
        self.decisionMaker = decisionClass(modelParams, decisionMakerName, fullDirectoryName, loadFiles, isMultiThreaded=isMultiThreaded)

        self.historyMngr = HistoryMngr(modelParams, historyFileName, fullDirectoryName, isMultiThreaded, loadFiles)
        self.nonTrainingHistCount = 0

        if resultFileName != '':
            self.resultFile = ResultFile(fullDirectoryName + resultFileName, numTrials2Learn, loadFiles)
        else:
            self.resultFile = None

        if isMultiThreaded:
            self.endRunLock = Lock()
        else:
            self.endRunLock = EmptyLock()

    def AddHistory(self):
        return self.historyMngr.AddHistory()

    def ExploreProb(self):
        return self.decisionMaker.ExploreProb()



    def choose_action(self, state):
        return self.decisionMaker.choose_action(self.historyMngr.NormalizeState(state))     

    def NumRuns(self):
        return self.decisionMaker.NumRuns()

    def TrimHistory(self):
        count = self.nonTrainingHistCount + 1
        if count % self.numTrials2Learn == 0:
            self.historyMngr.TrimHistory()
            print("\t", threading.current_thread().getName(), ": Trim History")

        self.nonTrainingHistCount += 1

    def end_run(self, r, score, steps):
        self.endRunLock.acquire()

        numRun = int(self.NumRuns())
        print(threading.current_thread().getName(), ": for trial#", numRun, ": reward =", r, "score =", score, "steps =", steps)

        learnAndSave = False
        if (numRun + 1) % self.numTrials2Learn == 0:
            learnAndSave = True

        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, learnAndSave)

        if learnAndSave:
            self.trial2LearnDQN = numRun + 1
            self.trainFlag = True 
                
        self.decisionMaker.end_run(r, learnAndSave)

        self.endRunLock.release()
        
        return learnAndSave 

    def Train(self):
        s,a,rVec,s_, terminal = self.historyMngr.GetHistory()
        
        if len(a) > self.params.minReplaySize:
            start = datetime.datetime.now()
            
            self.decisionMaker.learn(s,a,rVec,s_, terminal)
            self.endRunLock.acquire()
            self.decisionMaker.SaveDQN(self.trial2LearnDQN)
            self.endRunLock.release()

            diff = datetime.datetime.now() - start
            msDiff = diff.seconds * 1000 + diff.microseconds / 1000
            
            print("\t", threading.current_thread().getName(), ": ExperienceReplay - training with hist size = ", len(rVec), ", last", msDiff, "milliseconds")
        else:
            print("\t", threading.current_thread().getName(), ": ExperienceReplay size to small - training with hist size = ", len(rVec))


    def ResetAllData(self):
        self.decisionMaker.Reset()
        self.historyMngr.Reset()

        if self.resultFile != None:
            self.resultFile.Reset()

    def ActionValuesVec(self, s, targetValues = False):
        s = self.historyMngr.NormalizeState(s)
        return self.decisionMaker.ActionValuesVec(s, targetValues)

    def DiscountFactor(self):
        return self.decisionMaker.DiscountFactor()
