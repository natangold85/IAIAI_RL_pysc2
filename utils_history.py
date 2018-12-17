import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys

from multiprocessing import Lock
from utils import EmptyLock

def GetHistoryFromFile(name):
    transitions = None
    if os.path.isfile(name + '.gz') and os.path.getsize(name + '.gz') > 0:
        transitions = pd.read_pickle(name + '.gz', compression='gzip')

    return transitions

def JoinTransitions(target, toJoin):
    keyTransitions = ["s", "a", "r", "s_", "terminal"]
    for k in keyTransitions:
        if k in toJoin.keys():
            if k in target.keys():
                target[k] += toJoin[k]
            else:
                target[k] = toJoin [k]

    if "maxStateVals" in toJoin.keys():
        if "maxStateVals" in target.keys():
            target["maxStateVals"] = np.maximum(target["maxStateVals"], toJoin["maxStateVals"])
        else:
            target["maxStateVals"] = toJoin["maxStateVals"]

    if "rewardMax" in toJoin.keys():
        if "rewardMax" in target.keys():
            target["rewardMax"] = max(target["rewardMax"], toJoin["rewardMax"])
        else:
            target["rewardMax"] = toJoin["rewardMax"]

    if "rewardMin" in toJoin.keys():
        if "rewardMin" in target.keys():
            target["rewardMin"] = min(target["rewardMin"], toJoin["rewardMin"])
        else:
            target["rewardMin"] = toJoin["rewardMin"]

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

    def ExtractHistory(self, transitions):                    
        s = np.array(transitions["s"], dtype = float)
        a = np.array(transitions["a"], dtype = int)
        r = np.array(transitions["r"], dtype = float)
        s_ = np.array(transitions["s_"], dtype = float)
        terminal = np.array(transitions["terminal"], dtype = bool)

        return s, a, r, s_, terminal

class HistoryMngr(History):
    def __init__(self, params, historyFileName = '', directory = '', isMultiThreaded = False):
        super(HistoryMngr, self).__init__(isMultiThreaded)

        self.params = params

        self.isMultiThreaded = isMultiThreaded

        self.metaDataFields = ["maxStateVals", "rewardMax", "rewardMin"]

        self.transitions["maxStateVals"] = np.ones(params.stateSize, int)
        self.transitions["rewardMax"] = 0.01
        self.transitions["rewardMin"] = 0.0


        if self.params.saveOldHist:
            self.oldTransitions = {}
            for key in self.transitionKeys:
                self.oldTransitions[key] = []

        self.historyData = []

        self.trimmingHistory = False

        if historyFileName != '':
            self.histFileName = directory + historyFileName
            if self.params.saveOldHist:
                self.lastHistFileAdd = "_last"
                self.numOldHistFiles = 0
        else:
            self.histFileName = historyFileName
        
    def Load(self):
        if os.path.isfile(self.histFileName + '.gz') and os.path.getsize(self.histFileName + '.gz') > 0:
            self.transitions = pd.read_pickle(self.histFileName + '.gz', compression='gzip')

        if self.params.saveOldHist:
            if os.path.isfile(self.histFileName + self.lastHistFileAdd + '.gz'):
                self.oldTransitions = pd.read_pickle(self.histFileName + self.lastHistFileAdd + '.gz', compression='gzip')

            while os.path.isfile(self.histFileName + str(self.numOldHistFiles) +'.gz'):
                self.numOldHistFiles += 1

    def AddHistory(self):
        history = History(self.isMultiThreaded)
        self.historyData.append(history)
        return history

    def JoinHistoryFromSons(self):
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
        if self.params.saveOldHist:
            for key in self.transitionKeys:
                self.oldTransitions[key] += self.transitions[key][:toCut]
 
        for key in self.transitionKeys:
            del self.transitions[key][:toCut]
    
    def GetSingleHistory(self, history):
        transitions = history.GetHistory(reset=True)
        for key in self.transitionKeys:
            self.transitions[key] += transitions[key]

    def CleanHistory(self):
        for key in self.transitionKeys:
            self.transitions[key] = []

    def GetHistory(self, singleHist=None, shuffle=True):   
        self.histLock.acquire()

        if singleHist == None:
            self.JoinHistoryFromSons()
        else:
            self.GetSingleHistory(singleHist)


        self.PopHist2ReplaySize()
        allTransitions = self.transitions.copy()
        
        if not self.params.accumulateHistory:
            self.CleanHistory()

        self.histLock.release()

        
        if len(allTransitions["r"]) == 0:
            emptyM = np.array([]) 
            return emptyM, emptyM, emptyM, emptyM, emptyM

        s, a, r, s_, terminal = self.ExtractHistory(allTransitions)
        
        # normalization of transition values
        s, s_ = self.NormalizeStateVals(s, s_, allTransitions)
        if self.params.normalizeRewards:
            r = self.NormalizeRewards(r)

        if self.params.numRepeatsTerminalLearning > 0:
            s, a, r, s_, terminal = self.AddTerminalStates(s, a, r, s_, terminal)            
        
        self.SaveHistFile(allTransitions)    

        size = len(a)
        idx4Shuffle = np.arange(size)
        
        if shuffle:
            np.random.shuffle(idx4Shuffle)

        return s[idx4Shuffle], a[idx4Shuffle], r[idx4Shuffle], s_[idx4Shuffle], terminal[idx4Shuffle]
    
    def AddMetaDataFields2Dict(self, transitions):
        for key in self.metaDataFields:
            transitions[key] = self.transitions[key]

    def TrimHistory(self):
        self.histLock.acquire()
        if not self.trimmingHistory:
            self.trimmingHistory = True
            
            self.JoinHistoryFromSons()
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

        s = s / transitions["maxStateVals"] 
        s_ = s_ / transitions["maxStateVals"] 

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
    

    def Save(self):
        self.histLock.acquire()
        self.JoinHistoryFromSons()
        pd.to_pickle(self.transitions, self.histFileName + '.gz', 'gzip') 
        self.histLock.release()

    def SaveHistFile(self, transitions):
        if self.histFileName != '':
            pd.to_pickle(transitions, self.histFileName + '.gz', 'gzip') 
            if os.path.getsize(self.histFileName + '.gz') == 0:
                print("\n\n\n\nError save 0 bytes\n\n\n")
            
            if self.params.saveOldHist:
                if len(self.oldTransitions["a"]) >= self.params.maxReplaySize:
                    pd.to_pickle(self.oldTransitions, self.histFileName + str(self.numOldHistFiles) + '.gz', 'gzip') 
                    
                    self.numOldHistFiles += 1
                    
                    for key in self.transitionKeys:
                        self.oldTransitions[key] = []
                elif len(self.oldTransitions["a"]) > 0:
                    pd.to_pickle(self.oldTransitions, self.histFileName + self.lastHistFileAdd + '.gz', 'gzip') 

    def DrawState(self, realState):
        if realState:
            sizeHist = len(self.transitions["a"])
            if sizeHist > 0:
                idx = np.random.randint(0,sizeHist)
                s = self.transitions["s"][idx].copy()
            else:
                s = np.array([])
        else:
            s = self.transitions["maxStateVals"].copy()
            if max(s) == 1:
                s = np.array([])
            else:
                for i in range(len(s)):
                    s[i] = np.random.randint(0, s[i])
            
        return s


    def GetAllHist(self):
        transitions = {}
        for key in self.transitionKeys:
            transitions[key] = []

        if self.params.saveOldHist:
            for i in range(self.numOldHistFiles):
                currTransitions = pd.read_pickle(self.histFileName + str(i) + '.gz', compression='gzip')
                for key in self.transitionKeys:
                    transitions[key] += currTransitions[key]
            
            for key in self.transitionKeys:
                transitions[key] += self.oldTransitions[key] 
        
        for key in self.transitionKeys:
            transitions[key] += self.transitions[key]   

        return transitions

    def Reset(self, dump2Old=True, save=False):
        self.histLock.acquire()
        if dump2Old:
            while (len(self.transitions["a"]) > 0):
                for key in self.transitionKeys:
                    self.oldTransitions[key].append(self.transitions[key].pop(0))
        else:
            for key in self.transitionKeys:
                self.transitions[key] = []
        
        if save:
            self.SaveHistFile(self.transitions)

        self.histLock.release()

    def GetTransitionsSortedByIdx(self, idx):
        self.histLock.acquire()
        s = np.array(self.transitions["s"])
        a = np.array(self.transitions["a"])
        r = np.array(self.transitions["r"])
        s_ = np.array(self.transitions["s_"])
        terminal = np.array(self.transitions["terminal"])
        self.histLock.release()

        sortedIdx = s[:, idx].argsort()



        return s[sortedIdx,:], a[sortedIdx], r[sortedIdx], s_[sortedIdx,:], terminal[sortedIdx]