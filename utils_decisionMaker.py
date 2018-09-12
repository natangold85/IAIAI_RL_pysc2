import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys

from multiprocessing import Lock

from utils import EmptyLock

#decision makers
from utils_dqn import DQN
from utils_dqn import DQN_WithTarget

from utils_qtable import QLearningTable

# prev models
# from hallucination import HallucinationMngrPSFunc

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

        if directory != "":
            fullDirectoryName = "./" + directory +"/"
            if not os.path.isdir(fullDirectoryName):
                os.makedirs(fullDirectoryName)
        else:
            fullDirectoryName = "./"
        
        self.numTrials2Learn = numTrials2Learn
        self.trialNum = 0
        self.trial2LearnDQN = -1

        self.params = modelParams

        self.nextExperienceIdx2Add = 0
        self.transitions = {}
        self.transitions["s"] = []
        self.transitions["a"] = []
        self.transitions["r"] = []
        self.transitions["s_"] = []
        self.transitions["terminal"] = []

        self.transitions["maxStateVals"] = np.ones(modelParams.stateSize, int)

        if "new" in sys.argv:
            loadFiles = False
        else:
            loadFiles = True

        decisionClass = eval(modelType)
        self.decisionMaker = decisionClass(modelParams, decisionMakerName, fullDirectoryName, loadFiles, isMultiThreaded=isMultiThreaded)

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

        if isMultiThreaded:
            self.learnLock = Lock()
            self.endRunLock = Lock()
        else:
            self.endRunLock = EmptyLock()
            self.learnLock = EmptyLock()

    def ExploreProb(self):
        return self.decisionMaker.ExploreProb()

    def normalizeState(self, state):
        return state / self.transitions["maxStateVals"]

    def choose_action(self, observation):
        observation = self.normalizeState(observation)
        return self.decisionMaker.choose_action(observation)     

    def NumRuns(self):
        return self.decisionMaker.NumRuns()

    def learn(self, s, a, r, s_, terminal = False):

        if len(self.transitions["a"]) < self.params.maxReplaySize:
            self.transitions["s"].append(s.copy())
            self.transitions["a"].append(a)
            self.transitions["r"].append(r)
            self.transitions["s_"].append(s_.copy())
            self.transitions["terminal"].append(terminal)
        else:
            self.learnLock.acquire()
            idx = self.nextExperienceIdx2Add
            self.nextExperienceIdx2Add = (idx + 1) % self.params.maxReplaySize
            self.learnLock.release()

            self.transitions["s"][idx] = s.copy()
            self.transitions["a"][idx] = a
            self.transitions["r"][idx] = r
            self.transitions["s_"][idx] = s_.copy()
            self.transitions["terminal"][idx] = terminal
        

    def NormalizeStateVals(self, s, s_):
        maxAll = np.column_stack((self.transitions["maxStateVals"], np.max(s, axis = 0), np.max(s_, axis = 0)))
        self.transitions["maxStateVals"] = np.max(maxAll, axis = 1)

        s /= self.transitions["maxStateVals"]
        s_ /= self.transitions["maxStateVals"]

        return s , s_

    def ExperienceReplay(self):   
        idx4Shuffle = np.arange(len(self.transitions["r"]))
        np.random.shuffle(idx4Shuffle)
        size = int(len(idx4Shuffle) * self.params.historyProportion4Learn)
        chosenIdx = idx4Shuffle[0:size]
                
        s = np.array(self.transitions["s"], dtype = float)[chosenIdx]
        a = np.array(self.transitions["a"], dtype = int)[chosenIdx]
        r = np.array(self.transitions["r"], dtype = float)[chosenIdx]
        s_ = np.array(self.transitions["s_"], dtype = float)[chosenIdx]
        terminal = np.array(self.transitions["terminal"], dtype = bool)[chosenIdx]

        s, s_ = self.NormalizeStateVals(s, s_)

        if self.histFileName != '':
            pd.to_pickle(self.transitions, self.histFileName, 'gzip') 

        return s, a, r, s_, terminal

    def end_run(self, r, score, steps):
        self.endRunLock.acquire()

        numRun = int(self.NumRuns())
        print("\t\tfor trial#", numRun, ": reward =", r, "score =", score, "steps =", steps)

        learnAndSave = False
        self.trialNum += 1

        if self.trialNum % self.numTrials2Learn == 0:
            learnAndSave = True

        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, learnAndSave)

        if self.tTable != None:
            self.tTable.end_run(learnAndSave)

        if learnAndSave:
            self.trial2LearnDQN = numRun + 1
            self.trainFlag = True 
                
        self.decisionMaker.end_run(r, learnAndSave)

        self.endRunLock.release()
        
        return learnAndSave 

    def Train(self):
        s,a,rVec,s_, terminal = self.ExperienceReplay()
        
        if len(a) > self.params.minReplaySize:
            start = datetime.datetime.now()
            
            self.decisionMaker.learn(s,a,rVec,s_, terminal)
            self.endRunLock.acquire()
            self.decisionMaker.SaveDQN(self.trial2LearnDQN)
            self.endRunLock.release()

            diff = datetime.datetime.now() - start
            msDiff = diff.seconds * 1000 + diff.microseconds / 1000
            
            print("ExperienceReplay - training with hist size = ", len(rVec), ", training last", msDiff, "milliseconds")


    def ResetAllData(self):
        self.transitions = {}
        self.transitions["s"] = []
        self.transitions["a"] = []
        self.transitions["r"] = []
        self.transitions["s_"] = []
        self.transitions["terminal"] = []

        self.decisionMaker.Reset()

        if self.resultFile != None:
            self.resultFile.Reset()

    def ActionValuesVec(self, s, targetValues = False):
        s = self.normalizeState(s)
        return self.decisionMaker.ActionValuesVec(s, targetValues)

    def DiscountFactor(self):
        return self.decisionMaker.DiscountFactor()
