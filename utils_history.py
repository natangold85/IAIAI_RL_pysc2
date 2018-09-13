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
        self.isMultiThreaded = isMultiThreaded

        self.transitions["maxStateVals"] = np.ones(params.stateSize, int)

        self.oldTransitions = {}
        for key in self.transitionKeys:
            self.oldTransitions[key] = []

        self.historyData = []

        self.trimmingHistory = False

        if historyFileName != '':
            self.addition2CurrHistFile = "_curr"
            self.histFileName = directory + historyFileName
            self.numOldHistFiles = 0
            
            if loadFiles:
                if os.path.isfile(self.histFileName + self.addition2CurrHistFile + '.gz'):
                    self.transitions = pd.read_pickle(self.histFileName + self.addition2CurrHistFile + '.gz', compression='gzip')

                if os.path.isfile(self.histFileName + '.gz'):
                    self.oldTransitions = pd.read_pickle(self.histFileName + '.gz', compression='gzip')

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
        return state / self.transitions["maxStateVals"]

    def NormalizeStateVals(self, s, s_):
        maxAll = np.column_stack((self.transitions["maxStateVals"].copy(), np.max(s, axis = 0), np.max(s_, axis = 0)))
        self.transitions["maxStateVals"] = np.max(maxAll, axis = 1)

        maxStateVals = self.transitions["maxStateVals"]
        
        s /= maxStateVals
        s_ /= maxStateVals

        return s , s_

    def SaveHistFile(self, transitions):

        if self.histFileName != '':
            pd.to_pickle(transitions, self.histFileName + self.addition2CurrHistFile + '.gz', 'gzip') 
            
            if len(self.oldTransitions["a"]) >= self.maxReplaySize:
                pd.to_pickle(self.oldTransitions, self.histFileName + str(self.numOldHistFiles) + '.gz', 'gzip') 
                
                self.numOldHistFiles += 1
                
                for key in self.transitionKeys:
                    self.oldTransitions[key] = []
            else:
                pd.to_pickle(self.oldTransitions, self.histFileName + '.gz', 'gzip') 