import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys

from multiprocessing import Lock
from utils import EmptyLock


class History():
    def __init__(self, isMultiThreaded = False):

        self.transitions = {}
        self.transitionKeys = ["s", "a", "r", "s_", "terminal"]

        for key in self.transitionKeys:
            self.transitions[key] = []

        #self.transitions["maxStateVals"] = np.ones(modelParams.stateSize, int)
        if isMultiThreaded:
            self.histLock = Lock()
        else:
            self.histLock = EmptyLock()


    def learn(self, s, a, r, s_, terminal = False):
        self.histLock.acquire()
        
        self.transitions["s"].append(s.copy())
        self.transitions["a"].append(a)
        self.transitions["r"].append(r)
        self.transitions["s_"].append(s_.copy())
        self.transitions["terminal"].append(terminal)

        self.histLock.release()
        
    def GetHistory(self, reset = True):
        self.histLock.acquire()
        transitions = self.transitions.copy()
        if reset:
            for key in self.transitionKeys:
                self.transitions[key] = []
        self.histLock.release()

        return transitions


    def Reset(self):
        self.histLock.acquire()
        for key in self.transitionKeys:
            self.transitions[key] = []
        self.histLock.release()


class HistoryMngr(History):
    def __init__(self, params, historyFileName = '', directory = '', isMultiThreaded = False, loadFiles = True):
        super(HistoryMngr, self).__init__(isMultiThreaded)

        self.maxReplaySize = params.maxReplaySize
        self.normalizeRewards = params.normalizeRewards

        self.isMultiThreaded = isMultiThreaded

        self.transitions["maxStateVals"] = np.ones(params.stateSize, int)
        self.transitions["rewardMax"] = 0.0
        self.transitions["rewardMin"] = 0.0

        self.oldTransitions = {}
        for key in self.transitionKeys:
            self.oldTransitions[key] = []

        self.historyData = []

        self.trimmingHistory = False

        if historyFileName != '':
            self.lastHistFile = "_last"
            self.histFileName = directory + historyFileName
            self.numOldHistFiles = 0
            
            if loadFiles:
                if os.path.isfile(self.histFileName + '.gz'):
                    self.transitions = pd.read_pickle(self.histFileName + '.gz', compression='gzip')

                if os.path.isfile(self.histFileName + self.lastHistFile + '.gz'):
                    self.oldTransitions = pd.read_pickle(self.histFileName + self.lastHistFile + '.gz', compression='gzip')

                while os.path.isfile(self.histFileName + str(self.numOldHistFiles) +'.gz'):
                    self.numOldHistFiles += 1
        else:
            self.histFileName = historyFileName


    def AddHistory(self):
        history = History(self.isMultiThreaded)
        self.historyData.append(history)
        return history

    def JoinHistroyFromSons(self):
        size = 0
        for hist in self.historyData:
            transitions = hist.GetHistory()
            size += len(transitions["a"])
            for key in self.transitionKeys:
                self.transitions[key] += transitions[key]

        return size

    def Size(self):
        return len(self.transitions["a"])       
     
    def GetHistory(self):   
        self.histLock.acquire()

        self.JoinHistroyFromSons()

        while (len(self.transitions["a"]) > self.maxReplaySize):
            for key in self.transitionKeys:
                self.oldTransitions[key].append(self.transitions[key].pop(0))

        allTransitions = self.transitions.copy()
        self.histLock.release()


        idx4Shuffle = np.arange(len(allTransitions["r"]))
        np.random.shuffle(idx4Shuffle)
        
        s = np.array(allTransitions["s"], dtype = float)[idx4Shuffle]
        a = np.array(allTransitions["a"], dtype = int)[idx4Shuffle]
        r = np.array(allTransitions["r"], dtype = float)[idx4Shuffle]
        s_ = np.array(allTransitions["s_"], dtype = float)[idx4Shuffle]
        terminal = np.array(allTransitions["terminal"], dtype = bool)[idx4Shuffle]

        s, s_ = self.NormalizeStateVals(s, s_)
        if self.normalizeRewards:
            r = self.NormalizeRewards(r)

        self.SaveHistFile(allTransitions)    

        return s, a, r, s_, terminal
    
    def TrimHistory(self):
        self.histLock.acquire()
        if not self.trimmingHistory:
            self.trimmingHistory = True
            
            self.JoinHistroyFromSons()
            
            while (len(self.transitions["a"]) > self.maxReplaySize):
                for key in self.transitionKeys:
                    self.oldTransitions[key].append(self.transitions[key].pop(0))
            
            transitions = self.transitions.copy()
            self.histLock.release()

            self.SaveHistFile(transitions) 
            self.trimmingHistory = False
        else:
            self.histLock.release()

    def NormalizeState(self, state):
        return (state * 2) / self.transitions["maxStateVals"] - 1.0

    def NormalizeStateVals(self, s, s_):
        maxAll = np.column_stack((self.transitions["maxStateVals"].copy(), np.max(s, axis = 0), np.max(s_, axis = 0)))
        self.transitions["maxStateVals"] = np.max(maxAll, axis = 1)

        maxStateVals = self.transitions["maxStateVals"]
        
        s = (s * 2) / maxStateVals - 1.0
        s_ = (s_ * 2) / maxStateVals - 1.0

        return s , s_
    
    def NormalizeRewards(self, r):
        self.transitions["rewardMax"] = max(self.transitions["rewardMax"] , np.max(r))
        self.transitions["rewardMin"] = min(self.transitions["rewardMin"] , np.min(r))
        midReward = self.transitions["rewardMax"] - self.transitions["rewardMin"]
        return r - midReward


    def SaveHistFile(self, transitions):

        if self.histFileName != '':
            pd.to_pickle(transitions, self.histFileName + '.gz', 'gzip') 
            
            if len(self.oldTransitions["a"]) >= self.maxReplaySize:
                pd.to_pickle(self.oldTransitions, self.histFileName + str(self.numOldHistFiles) + '.gz', 'gzip') 
                
                self.numOldHistFiles += 1
                
                for key in self.transitionKeys:
                    self.oldTransitions[key] = []
            else:
                pd.to_pickle(self.oldTransitions, self.histFileName + self.lastHistFile + '.gz', 'gzip') 

    def DrawState(self):
        sizeHist = len(self.transitions["a"])
        if sizeHist > 0:
            idx = np.random.randint(0,sizeHist)
            return self.transitions["s"][idx]       
        else:
            return None

    def GetAllHist(self):
        transitions = {}
        for key in self.transitionKeys:
            transitions[key] = []

        for i in range(self.numOldHistFiles):
           currTransitions = pd.read_pickle(self.histFileName + str(i) + '.gz', compression='gzip')
           for key in self.transitionKeys:
               transitions[key] += currTransitions[key]

        for key in self.transitionKeys:
            transitions[key] += self.oldTransitions[key]  
            transitions[key] += self.transitions[key]   

        return transitions