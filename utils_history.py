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
    
    def RemoveNonTerminalHistory(self):
        self.histLock.acquire()
        idx = len(self.transitions["terminal"]) - 1
        while self.transitions["terminal"][idx] != True and idx >= 0:
            for key in self.transitionKeys:
                self.transitions[key].pop(-1)
            idx -= 1
        self.histLock.release()



class HistoryMngr(History):
    def __init__(self, params, historyFileName = '', directory = '', isMultiThreaded = False, loadFiles = True, createAllHistFiles = True):
        super(HistoryMngr, self).__init__(isMultiThreaded)

        self.params = params

        self.isMultiThreaded = isMultiThreaded

        self.transitions["maxStateVals"] = np.ones(params.stateSize, int)
        self.transitions["rewardMax"] = 0.01
        self.transitions["rewardMin"] = 0.0

        self.createAllHistFiles = createAllHistFiles

        if createAllHistFiles:
            self.oldTransitions = {}
            for key in self.transitionKeys:
                self.oldTransitions[key] = []

        self.historyData = []

        self.trimmingHistory = False

        if historyFileName != '':
            self.histFileName = directory + historyFileName
            
            if loadFiles:
                if os.path.isfile(self.histFileName + '.gz'):
                    self.transitions = pd.read_pickle(self.histFileName + '.gz', compression='gzip')


            if self.createAllHistFiles:
                self.lastHistFileAdd = "_last"
                self.numOldHistFiles = 0

                if loadFiles:
                    if os.path.isfile(self.histFileName + self.lastHistFileAdd + '.gz'):
                        self.oldTransitions = pd.read_pickle(self.histFileName + self.lastHistFileAdd + '.gz', compression='gzip')

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
    
    def PopHist2ReplaySize(self):
        toCut = len(self.transitions["a"]) - self.params.maxReplaySize
        if self.createAllHistFiles:
            for key in self.transitionKeys:
                self.oldTransitions[key] += self.transitions[key][:toCut]
 
        for key in self.transitionKeys:
            del self.transitions[key][:toCut]
                            
    def GetHistory(self):   
        self.histLock.acquire()

        self.JoinHistroyFromSons()
        self.PopHist2ReplaySize()

        allTransitions = self.transitions.copy()
        self.histLock.release()

        if len(allTransitions["r"]) == 0:
            emptyM = np.array([]) 
            return emptyM, emptyM, emptyM, emptyM, emptyM

        
        s = np.array(allTransitions["s"], dtype = float)
        a = np.array(allTransitions["a"], dtype = int)
        r = np.array(allTransitions["r"], dtype = float)
        s_ = np.array(allTransitions["s_"], dtype = float)
        terminal = np.array(allTransitions["terminal"], dtype = bool)

        # normalization of transition values
        s, s_ = self.NormalizeStateVals(s, s_, allTransitions)
        if self.params.normalizeRewards:
            r = self.NormalizeRewards(r)

        if self.params.numRepeatsTerminalLearning > 0:
            s, a, r, s_, terminal = self.AddTerminalStates(s, a, r, s_, terminal)            
        
        self.SaveHistFile(allTransitions)    

        size = len(a)
        idx4Shuffle = np.arange(size)
        np.random.shuffle(idx4Shuffle)
        return s[idx4Shuffle], a[idx4Shuffle], r[idx4Shuffle], s_[idx4Shuffle], terminal[idx4Shuffle]
    
    def TrimHistory(self):
        self.histLock.acquire()
        if not self.trimmingHistory:
            self.trimmingHistory = True
            
            self.JoinHistroyFromSons()
            self.PopHist2ReplaySize()
            
            transitions = self.transitions.copy()
            self.histLock.release()

            if len(transitions["a"]) > 0:
                s = np.array(transitions["s"])
                s_ = np.array(transitions["s_"])
                self.FindMaxStateVals(s, s_, transitions)

            self.SaveHistFile(transitions) 
            self.trimmingHistory = False
        else:
            self.histLock.release()

    def NormalizeState(self, state):
        return (state * 2) / self.transitions["maxStateVals"] - 1.0

    def FindMaxStateVals(self, s, s_, transitions):
        maxAll = np.column_stack((transitions["maxStateVals"], np.max(s, axis = 0), np.max(s_, axis = 0)))
        transitions["maxStateVals"] = np.max(maxAll, axis = 1)
        
        self.transitions["maxStateVals"] = transitions["maxStateVals"].copy()

    def NormalizeStateVals(self, s, s_, transitions):
        
        self.FindMaxStateVals(s, s_, transitions)

        s = (s * 2) / transitions["maxStateVals"] - 1.0
        s_ = (s_ * 2) / transitions["maxStateVals"] - 1.0

        return s , s_
    
    def NormalizeRewards(self, r):
        self.transitions["rewardMax"] = max(self.transitions["rewardMax"] , np.max(r))
        self.transitions["rewardMin"] = min(self.transitions["rewardMin"] , np.min(r))

        return (r - self.transitions["rewardMin"]) / (self.transitions["rewardMax"] - self.transitions["rewardMin"])

    def AddTerminalStates(self, s, a, r, s_, terminal):
        terminalIdx = terminal.nonzero()
        np.set_printoptions(threshold=np.nan)
        

        sT = np.repeat(np.squeeze(s[terminalIdx, :]), self.params.numRepeatsTerminalLearning, axis=0)
        aT = np.repeat(np.squeeze(a[terminalIdx]), self.params.numRepeatsTerminalLearning)
        rT = np.repeat(np.squeeze(r[terminalIdx]), self.params.numRepeatsTerminalLearning)
        s_T = np.repeat(np.squeeze(s_[terminalIdx, :]), self.params.numRepeatsTerminalLearning, axis=0)
        terminalT = np.ones(len(aT), dtype=bool)

        s = np.concatenate((s, sT))
        a = np.concatenate((a, aT))
        r = np.concatenate((r, rT))
        s_ = np.concatenate((s_, s_T))
        terminal = np.concatenate((terminal, terminalT))
        return s, a, r, s_, terminal 
            

    def GetMinReward(self):
        return self.transitions["rewardMin"]

    def GetMaxReward(self):
        return self.transitions["rewardMax"]

    def SetMinReward(self, r):
        self.transitions["rewardMin"] = min(self.transitions["rewardMin"], r)
    
    def SetMaxReward(self, r):
        self.transitions["rewardMax"] = max(self.transitions["rewardMax"], r)

    def SaveHistFile(self, transitions):

        if self.histFileName != '':
            if len(transitions["a"]) > 0:
                pd.to_pickle(transitions, self.histFileName + '.gz', 'gzip') 
                if os.path.getsize(self.histFileName + '.gz') == 0:
                    print("\n\n\n\nError save 0 bytes\n\n\n")
            
            if self.createAllHistFiles:
                if len(self.oldTransitions["a"]) >= self.params.maxReplaySize:
                    pd.to_pickle(self.oldTransitions, self.histFileName + str(self.numOldHistFiles) + '.gz', 'gzip') 
                    
                    self.numOldHistFiles += 1
                    
                    for key in self.transitionKeys:
                        self.oldTransitions[key] = []
                elif len(self.oldTransitions["a"]) > 0:
                    pd.to_pickle(self.oldTransitions, self.histFileName + self.lastHistFileAdd + '.gz', 'gzip') 

    def DrawState(self):
        sizeHist = len(self.transitions["a"])
        if sizeHist > 0:
            idx = np.random.randint(0,sizeHist)
            return self.transitions["s"][idx]       
        else:
            return np.array([])

    def GetAllHist(self):
        transitions = {}
        for key in self.transitionKeys:
            transitions[key] = []

        if self.createAllHistFiles:
            for i in range(self.numOldHistFiles):
                currTransitions = pd.read_pickle(self.histFileName + str(i) + '.gz', compression='gzip')
                for key in self.transitionKeys:
                    transitions[key] += currTransitions[key]
            
            for key in self.transitionKeys:
                transitions[key] += self.oldTransitions[key] 
        
        for key in self.transitionKeys:
            transitions[key] += self.transitions[key]   

        return transitions

    def Reset(self, dump2Old=True):
        self.histLock.acquire()
        if dump2Old:
            while (len(self.transitions["a"]) > 0):
                for key in self.transitionKeys:
                    self.oldTransitions[key].append(self.transitions[key].pop(0))
        else:
            for key in self.transitionKeys:
                self.transitions[key] = []
        
        self.histLock.release()