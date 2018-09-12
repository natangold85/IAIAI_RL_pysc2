import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from utils_dqn import DQN_PARAMS
from utils_dqn import DQN

import random


def CreateDqn(useGpu = True):
    if not useGpu:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    dqnParams = DQN_PARAMS(16, 7, layersNum = 2)
    dqn = DQN(dqnParams, "test3", "./test3/", loadNN=False, isMultiThreaded=False)
    return dqn

def getHist():
    histFile = "replayHistory_buildOnly.gz"
    histDict = pd.read_pickle(histFile, compression='gzip')

    s = np.array(histDict["s"], dtype = float)
    a = np.array(histDict["a"], dtype = int)
    s_ = np.array(histDict["s_"], dtype = float)
    r = np.array(histDict["r"], dtype = float)
    terminal = np.array(histDict["terminal"], dtype = bool)

    s = NormalizeStateVals(histDict, s)
    s_ = NormalizeStateVals(histDict, s_)

    return s, a, r, s_, terminal


def train(dqn, histSize):
    s, a, r, s_, terminal = getHist()
    if len(a) < histSize:
        histSize = len(terminal)

    idxArray = list(range(len(a)))
    idxChosen = random.sample(idxArray, histSize)

    shuffS = s[idxChosen,:].reshape(histSize, s.shape[1])
    shuffA = a[idxChosen].reshape(histSize)
    shuffS_ = s_[idxChosen,:].reshape(histSize, s_.shape[1])
    shuffR = r[idxChosen].reshape(histSize)
    shuffT = terminal[idxChosen].reshape(histSize)

    dqn.learn(shuffS, shuffA, shuffR, shuffS_, shuffT)

def insertMax2Dict(histDict):
    histDict["maxStateVals"] = np.ones(16, int)

def NormalizeStateVals(histDict, s):
    return s / histDict["maxStateVals"]


def PrintValues(dqn, allStates):
    for i in range(allStates.shape[0]):
        values = dqn.ActionValuesVec(allStates[i, :])
        print("s =", allStates[i, :], "values =", end = ' [')
        for v in values:
            print(v, end = ', ')
        print("]")

# actions
    # ID_DO_NOTHING = 0
    # ID_BUILD_SUPPLY_DEPOT = 1
    # ID_BUILD_REFINERY = 2
    # ID_BUILD_BARRACKS = 3
    # ID_BUILD_FACTORY = 4
    # ID_BUILD_BARRACKS_REACTOR = 5
    # ID_BUILD_FACTORY_TECHLAB = 6
    # NUM_ACTIONS = 7

# check States
    # COMMAND_CENTER_IDX = 0
    # MINERALS_IDX = 1
    # GAS_IDX = 2
    # SUPPLY_DEPOT_IDX = 3
    # REFINERY_IDX = 4
    # BARRACKS_IDX = 5
    # FACTORY_IDX = 6
    # REACTORS_IDX = 7
    # TECHLAB_IDX = 8

    # IN_PROGRESS_SUPPLY_DEPOT_IDX = 9
    # IN_PROGRESS_REFINERY_IDX = 10
    # IN_PROGRESS_BARRACKS_IDX = 11
    # IN_PROGRESS_FACTORY_IDX = 12
    # IN_PROGRESS_REACTORS_IDX = 13
    # IN_PROGRESS_TECHLAB_IDX = 14    
    
    # SUPPLY_USED = 15
def getCheckStates():
    
    states =  np.array([[ 1, 0.5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # no sd (sd affect check)
       [ 1, 0.5,  0,  0.1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 1 sd (sd affect check)
       [ 1, 0.5,  0, 0.1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 7 sd (sd affect check)
       [ 1, 0.5,  0,  0.2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # no barracks (barracks affect check)
       [ 1, 0.5,  0,  0.2,  0,  0.33,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 1 barrack (barracks affect check)
       [ 1, 0.5,  0,  0.2,  0,  0.66,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 2 barracks (barracks affect check)
       [ 1, 0.5,  0,  0.2,  0,  0,  0,  0,  0,  0,  0,  0.5,  0,  0,  0,  0.2]]) # 1 barrack in progress (barracks affect check)

    return states




# from test3 import *
# dqn = CreateDqn()
# states = getCheckStates()
# PrintValues(dqn, states)

def MinMaxValues(dqn, states, size):
    sumMax = 0.0
    sumMin = 0.0
    for i in range(size):
        values = dqn.ActionValuesVec(states[i, :])
        sumMax += max(values)
        sumMin += min(values)    

    return [sumMax / size, sumMin / size]

def checkExploding(startSize = 32, endSize = 500000, numEpochs = 100, jumps = 32):
    results = {}

    checkStates = getCheckStates()
    checkSize = checkStates.shape[0]
    dqn = CreateDqn()
    
    for size in range(startSize, endSize, jumps):
        singleResults = []
        singleResults.append(MinMaxValues(dqn, checkStates, checkSize))
        
        for epoch in range(numEpochs):
            train(dqn, size)
        
        singleResults.append(MinMaxValues(dqn, checkStates, checkSize))
        results[size] = singleResults
        dqn.Reset()
        print("finished training size =", size)

    return results


        

