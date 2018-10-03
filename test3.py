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
