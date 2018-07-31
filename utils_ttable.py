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
from hallucination import HallucinationMngrPSFunc

import matplotlib.pyplot as plt

from multiprocessing import Process, Lock, Value, Array, Manager

from utils import ParamsBase

class TransitionTable:
    def __init__(self, numActions, tableName, newTable = False):
        self.tableName = tableName
        self.actions = list(range(numActions))  # a list

        self.table = {}
        
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        
        self.tableIdx = 0
        self.actionSumIdx = 1

        if os.path.isfile(tableName + '.gz') and not newTable:
            self.table = pd.read_pickle(tableName + '.gz', compression='gzip')
        else:
            self.table[self.TrialsData] = [0]

        self.numTotRuns = self.table[self.TrialsData][self.NumRunsTotalSlot]

    def check_item_exist(self, item):
        if item not in self.table:
            # append new state to q table
            self.table[item] = [None, None]
            self.table[item][self.tableIdx] = pd.DataFrame(columns=self.actions, dtype=np.float)
            self.table[item][self.actionSumIdx] = []
            for a in range(0, len(self.actions)):
                self.table[item][self.actionSumIdx].append(0)

    def check_state_exist(self, s, s_):
        self.check_item_exist(s)
        if s_ not in self.table[s][self.tableIdx].index:
            # append new state to q table
            self.table[s][self.tableIdx] = self.table[s][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.table[s][self.tableIdx].columns, name=s_))   

    def learn(self, s, a, s_):
        self.check_state_exist(s, s_)  

        # update transition
        self.table[s][self.tableIdx].ix[s_, a] += 1
        self.table[s][self.actionSumIdx][a] += 1

    def end_run(self, saveTable):
        self.numTotRuns += 1      
        if saveTable:
            self.table[self.TrialsData][self.NumRunsTotalSlot] = self.numTotRuns
            pd.to_pickle(self.table, self.tableName + '.gz', 'gzip') 
    
    def Reset(self):
        self.table = {}
        self.table[self.TrialsData] = [0]

        self.numTotRuns = self.table[self.TrialsData][self.NumRunsTotalSlot]

        
class BothWaysTransitionTable(TransitionTable):
    def __init__(self, numActions, tableName):
        super(BothWaysTransitionTable, self).__init__(numActions, tableName)
        self.normalKey = 0
        self.reverseKey = 1

        if self.normalKey not in self.table:
            self.table[self.normalKey] = {}

        if self.reverseKey not in self.table:
            self.table[self.reverseKey] = {}
            
    def check_item_exist(self, item, tableType):
        if item not in self.table[tableType]:
            # append new state to q table
            self.table[tableType][item] = [None, None]
            self.table[tableType][item][self.tableIdx] = pd.DataFrame(columns=self.actions, dtype=np.float)
            self.table[tableType][item][self.actionSumIdx] = [] #np.zeros(self.actions, dtype = int)
            for a in range(0, len(self.actions)):
                self.table[tableType][item][self.actionSumIdx].append(0)

    def check_state_exist(self, s, s_):
        self.check_item_exist(s, self.normalKey)
        self.check_item_exist(s_, self.reverseKey)
        
        if s_ not in self.table[self.normalKey][s][self.tableIdx].index:
            self.table[self.normalKey][s][self.tableIdx] = self.table[self.normalKey][s][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.table[self.normalKey][s][self.tableIdx].columns, name=s_))   

        if s not in self.table[self.reverseKey][s_][self.tableIdx].index:
            self.table[self.reverseKey][s_][self.tableIdx] = self.table[self.reverseKey][s_][self.tableIdx].append(pd.Series([0] * len(self.actions), index=self.table[self.reverseKey][s_][self.tableIdx].columns, name=s))   


    def learn(self, s, a, s_):

        self.check_state_exist(s, s_)  
        # update transition

        self.table[self.reverseKey][s_][self.actionSumIdx][a] += 1
        self.table[self.reverseKey][s_][self.tableIdx].ix[s, a] += 1

        self.table[self.normalKey][s][self.actionSumIdx][a] += 1
        self.table[self.normalKey][s][self.tableIdx].ix[s_, a] += 1