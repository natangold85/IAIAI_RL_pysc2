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
from utils_dqn import DQN_WithTargetAndDefault

from utils_qtable import QLearningTable

from utils_history import HistoryMngr

# model builders:
from utils_ttable import TransitionTable

# results handlers
from utils_results import ResultFile

class BaseDecisionMaker:
    def __init__(self, agentName):
        self.trainFlag = False
        self.switchFlag = False

        self.subAgentsDecisionMakers = {}
        self.switchCount = {}
        self.resultFile = None
        self.agentName = agentName

        self.copyTargetLock = Lock()
        self.copyTarget2DmNumRuns = -1

    def SetSubAgentDecisionMaker(self, key, decisionMaker):
        self.subAgentsDecisionMakers[key] = decisionMaker
        self.switchCount[key] = 0

    def GetSubAgentDecisionMaker(self, key):
        if key in self.subAgentsDecisionMakers.keys():
            return self.subAgentsDecisionMakers[key]
        else:
            return None

    def GetDecisionMakerByName(self, name):
        if self.agentName == name:
            return self
        else:
            for saDM in self.subAgentsDecisionMakers.values():
                if saDM != None:
                    dm = saDM.GetDecisionMakerByName(name)
                    if dm != None:
                        return dm
                
            return None
                

    def TrainAll(self):
        
        self.copyTargetLock.acquire()
        if self.copyTarget2DmNumRuns > 0:
            self.CopyTarget2Dm(self.copyTarget2DmNumRuns)
            self.copyTarget2DmNumRuns = -1
        self.copyTargetLock.release()


        if self.trainFlag:
            self.Train()
            self.trainFlag = False

        for subDM in self.subAgentsDecisionMakers.values():
            if subDM != None:
                subDM.TrainAll()

    def AddSwitch(self, idx, numSwitch, name, resultFile):
        if resultFile != None:
            if idx not in self.switchCount:
                self.switchCount[idx] = 0

            if self.switchCount[idx] <= numSwitch:
                self.switchCount[idx] = numSwitch + 1
                slotName = name + "_" + str(numSwitch)
                resultFile.AddSwitchSlot(slotName)

    def AddResultFile(self, resultFile):
        self.secondResultFile = resultFile
        
    def GetResultFile(self):
        return self.secondResultFile

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
    
    def DrawStateFromHist(self):
        return None
    
    def TakeTargetDM(self, numRunsEnd):
        self.copyTargetLock.acquire()
        self.copyTarget2DmNumRuns = numRunsEnd    
        self.copyTargetLock.release()

    def CopyTarget2Dm(self, numRuns):
        pass

    def GetMinReward(self):
        return 0.0
    
    def SetMinReward(self, r):
        pass

    def GetMaxReward(self):
        return 0.01
    
    def SetMaxReward(self, r):
        pass

    def IsWithDfltDecisionMaker(self):
        return False

    def NumDfltRuns(self):
        return 0

class BaseNaiveDecisionMaker(BaseDecisionMaker):
    def __init__(self, numTrials2Save, agentName = "", resultFName = None, directory = None):
        super(BaseNaiveDecisionMaker, self).__init__(agentName)
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

            self.resultFile = ResultFile(fullDirectoryName + resultFName, numTrials2Save, loadFiles, agentName)

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
    def __init__(self, agentName = "", playWithInput = True, numActions = 1, actionDoNothing = 0):
        super(UserPlay, self).__init__(agentName)
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
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', numTrials2Learn=100, isMultiThreaded = False):
        super(LearnWithReplayMngr, self).__init__(agentName)

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
        self.decisionMaker = decisionClass(modelParams, decisionMakerName, fullDirectoryName, loadFiles, isMultiThreaded=isMultiThreaded, agentName=self.agentName)

        self.historyMngr = HistoryMngr(modelParams, historyFileName, fullDirectoryName, isMultiThreaded, loadFiles)
        self.nonTrainingHistCount = 0

        if resultFileName != '':
            self.resultFile = ResultFile(fullDirectoryName + resultFileName, numTrials2Learn, loadFiles, self.agentName)
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

    def TargetExploreProb(self):
        return self.decisionMaker.TargetExploreProb()

    def choose_action(self, state):
        if not self.decisionMaker.TakeDfltValues():
            state = self.historyMngr.NormalizeState(state)

        return self.decisionMaker.choose_action(state)     

    def NumRuns(self):
        return self.decisionMaker.NumRuns()

    def TrimHistory(self):
        count = self.nonTrainingHistCount + 1
        if count % self.numTrials2Learn == 0:
            self.historyMngr.TrimHistory()
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->Trim History to size =", self.historyMngr.Size())

        self.nonTrainingHistCount += 1

    def end_run(self, r, score, steps):
        self.endRunLock.acquire()

        numRun = int(self.NumRuns())
        print(threading.current_thread().getName(), ":", self.agentName,"->for trial#", numRun, ": reward =", r, "score =", score, "steps =", steps)

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
        s,a,r,s_, terminal = self.historyMngr.GetHistory()
        
        if len(a) > self.params.minReplaySize:
            start = datetime.datetime.now()
            
            self.decisionMaker.learn(s, a, r, s_, terminal, self.trial2LearnDQN)
            self.endRunLock.acquire()
            self.decisionMaker.SaveDQN(self.trial2LearnDQN)
            self.endRunLock.release()

            diff = datetime.datetime.now() - start
            msDiff = diff.seconds * 1000 + diff.microseconds / 1000
            
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->ExperienceReplay - training with hist size = ", len(r), ", last", msDiff, "milliseconds")
        else:
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->ExperienceReplay size to small - training with hist size = ", len(r))


    def ResetHistory(self, dump2Old=True):
        self.historyMngr.Reset(dump2Old)

    def ResetAllData(self, resetDecisionMaker=True, resetHistory=True, resetResults=True):
        if resetDecisionMaker:
            self.decisionMaker.Reset()

        if resetHistory:
            self.historyMngr.Reset()

        if resetResults and self.resultFile != None:
            self.resultFile.Reset()


    def ActionValuesVec(self, s, targetValues = False):
        if not self.decisionMaker.TakeDfltValues():
            s = self.historyMngr.NormalizeState(s)

        return self.decisionMaker.ActionValuesVec(s, targetValues)

    def CopyTarget2Dm(self, numRuns):
        print("\t", threading.current_thread().getName(), ":", self.agentName,"->Copy Target 2 DQN")
        self.decisionMaker.CopyTarget2DQN(numRuns)
    
    def DiscountFactor(self):
        return self.decisionMaker.DiscountFactor()

    def DrawStateFromHist(self):
        return self.historyMngr.DrawState()

    def GetMinReward(self):
        return self.historyMngr.GetMinReward()
    
    def SetMinReward(self, r):
        self.historyMngr.SetMinReward(r)

    def GetMaxReward(self):
        return self.historyMngr.GetMaxReward()
    
    def SetMaxReward(self, r):
        self.historyMngr.SetMaxReward(r)

    def IsWithDfltDecisionMaker(self):
        return self.decisionMaker.IsWithDfltDecisionMaker()

    def NumDfltRuns(self):
        return self.decisionMaker.NumDfltRuns()

    def DfltValueInitialized(self):
        return self.decisionMaker.DfltValueInitialized()
        

