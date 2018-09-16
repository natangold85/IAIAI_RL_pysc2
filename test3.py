import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from utils_dqn import DQN_PARAMS
from utils_dqn import DQN

import random
import math

def CreateDqn(useGpu = False):
    if not useGpu:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # dqnParams = DQN_PARAMS(16, 7, layersNum = 2)
    dqnParams = DQN_PARAMS(11, 5, layersNum = 2)
    dqn = DQN(dqnParams, "test3", "./test3/", loadNN=False, isMultiThreaded=False)
    return dqn

def history():
    histFile = "replayHistory_trainOnly.gz"
    histDict = pd.read_pickle(histFile, compression='gzip')

    return histDict


def getHist(histDict):
    if histDict == None:
        histDict = history()

    s = np.array(histDict["s"], dtype = float)
    a = np.array(histDict["a"], dtype = int)
    s_ = np.array(histDict["s_"], dtype = float)
    r = np.array(histDict["r"], dtype = float)
    terminal = np.array(histDict["terminal"], dtype = bool)

    s = NormalizeStateVals(histDict, s)
    s_ = NormalizeStateVals(histDict, s_)

    return s, a, r, s_, terminal


def train(dqn, histSize, histDict):
    s, a, r, s_, terminal = getHist(histDict)
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

def NormalizeStateVals(histDict, s):
    return s / histDict["maxStateVals"]


def PrintValues(dqn, allStates):
    for i in range(allStates.shape[0]):
        values = dqn.ActionValuesVec(allStates[i, :])
        print("s =", allStates[i, :], "values =", end = ' [')
        for v in values:
            print(v, end = ', ')
        print("]")


## BUILDER:

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

## TRAINER:

    # MINERALS_IDX = 0
    # GAS_IDX = 1
    # BARRACKS_IDX = 2
    # FACTORY_IDX = 3
    # REACTORS_IDX = 4
    # TECHLAB_IDX = 5

    # SUPPLY_LEFT = 6

    # QUEUE_BARRACKS = 7
    # QUEUE_FACTORY = 8
    # QUEUE_FACTORY_WITH_TECHLAB = 9
    
    # ARMY_POWER = 10

    # SIZE = 11

def getCheckStates():
    
    # builder state:

    # states =  np.array([[ 1, 0.5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # no sd (sd affect check)
    #    [ 1, 0.5,  0,  0.1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 1 sd (sd affect check)
    #    [ 1, 0.5,  0, 0.1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 7 sd (sd affect check)
    #    [ 1, 0.5,  0,  0.2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # no barracks (barracks affect check)
    #    [ 1, 0.5,  0,  0.2,  0,  0.33,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 1 barrack (barracks affect check)
    #    [ 1, 0.5,  0,  0.2,  0,  0.66,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0.2],  # 2 barracks (barracks affect check)
    #    [ 1, 0.5,  0,  0.2,  0,  0,  0,  0,  0,  0,  0,  0.5,  0,  0,  0,  0.2]]) # 1 barrack in progress (barracks affect check)

    # trainer state:

    states =  np.array([[0.5,  0.3,  0.25,  0,  0,  0,  0,  0,  0,  0, 0],  # no sd (sd affect check)
       [0.5,  0.3,  0.25,  0,  0,  0,  0,  0,  0.5,  0, 0.2],  # 1 sd (sd affect check)
       [0.5,  0.3,  0.25,  0,  0,  0,  0,  0,  0.75,  0, 0.5],  # 7 sd (sd affect check)
       [0.5,  0.3,  0.25,  0.5,  0,  0,  0,  0,  0,  0, 0],  # no barracks (barracks affect check)
       [0.5,  0.3,  0.25,  0.5,  0,  0,  0,  0,  0,  0, 0.2],  # 1 barrack (barracks affect check)
       [0.5,  0.3,  0.25,  0.5,  0,  0,  0,  0,  0,  0, 0.5]]) # 1 barrack in progress (barracks affect check)

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

def checkExploding(size = 500000, numEpochs = 20, numLearns = 10):

    histDict = history()
    results = []

    checkStates = getCheckStates()
    checkSize = checkStates.shape[0]
    dqn = CreateDqn()
    
    for l in range(numLearns):
        singleResults = []
        singleResults.append(MinMaxValues(dqn, checkStates, checkSize))
        
        for epoch in range(numEpochs):
            train(dqn, size, histDict)
        
        singleResults.append(MinMaxValues(dqn, checkStates, checkSize))
        results.append(singleResults)
        dqn.Reset()
        print("finished training size =", size)

    return results
