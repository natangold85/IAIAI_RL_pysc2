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

class PlotMngr:
    def __init__(self, resultFilesNamesList = [], resultFilesDirectories = [], legendList = [], directory2Save = ""):
        self.resultFileList = []
        self.legendList = legendList

        self.scriptName = sys.argv[0]
        self.scriptName = self.scriptName.replace(".\\", "")
        self.scriptName = self.scriptName.replace(".py", "")

        for i in range(len(resultFilesNamesList)):
            if resultFilesDirectories[i] != '':
                name = './' + resultFilesDirectories[i] + '/' + resultFilesNamesList[i]
            else:
                name = resultFilesNamesList[i]

            resultFile = ResultFile(name)
            self.resultFileList.append(resultFile)

        if directory2Save != '':
            self.plotFName = './' + directory2Save + '/' + self.scriptName + "_resultsPlot"
        else:
            self.plotFName = self.scriptName + "_resultsPlot"

        for table in self.legendList:
            self.plotFName += "_" + table 
        
        self.plotFName += ".png"

    def ResultsFromTable(self, table, grouping, dataIdx):
        names = list(table.index)
        tableSize = len(names) -1
        results = np.zeros((2, tableSize), dtype  = float)
        subGroupingSizeAll = 0
        for name in names[:]:
            if name.isdigit():
                idx = int(name)
                currResult = table.ix[name, dataIdx]
                subGroupSize = table.ix[name, 0]
                results[0, idx] = subGroupSize
                results[1, idx] = currResult

                if subGroupSize > subGroupingSizeAll:
                    subGroupingSizeAll = subGroupSize
                if subGroupSize != subGroupingSizeAll:
                    print("\n\nerror in sub grouping size\n\n")
                    exit()
                
        groupSizes = int(math.ceil(grouping / subGroupingSizeAll))
        idxArray = np.arange(groupSizes)
        
        groupResults = []
        timeLine = []
        t = grouping
        for i in range(groupSizes - 1, tableSize):
            res = sum(results[1, idxArray]) / groupSizes 
            groupResults.append(res)
            timeLine.append(t)
            idxArray += 1
            t += subGroupingSizeAll
        

        return np.array(groupResults), np.array(timeLine)      

    def Plot(self, grouping):
        tableCol = ['count', 'reward', 'score', '# of steps']
        fig = plt.figure(figsize=(19.0, 11.0))
        fig.suptitle("results for " + self.scriptName + ":", fontsize=20)

        for idx in range(1, 4):
            plt.subplot(2,2,idx)  
            for table in self.resultFileList:
                results, t = self.ResultsFromTable(table.result_table, grouping, idx) 
                plt.plot(t, results)
                
            plt.ylabel('avg '+ tableCol[idx] + ' for ' + str(grouping) + ' trials')
            plt.xlabel('#trials')
            plt.title('Average ' + tableCol[idx])
            plt.grid(True)
            plt.legend(self.legendList, loc='best')

        # full screen
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        fig.savefig(self.plotFName)

class ResultFile:
    def __init__(self, tableName, numToWrite = 100, loadFile = True):
        self.saveFileName = tableName + '.gz'
                
        self.numToWrite = numToWrite

        self.rewardCol = list(range(4))
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(self.saveFileName) and loadFile:
            self.result_table = pd.read_pickle(self.saveFileName, compression='gzip')

        self.countCompleteKey = 'countComplete'
        self.check_state_exist(self.countCompleteKey)

        self.countIdx = 0
        self.rewardIdx = 1
        self.scoreIdx = 2
        self.stepsIdx = 3

        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])

        self.sumReward = 0
        self.sumScore = 0
        self.sumSteps = 0
        self.numRuns = 0

    def check_state_exist(self, state):
        if state not in self.result_table.index:
            # append new state to q table
            self.result_table = self.result_table.append(pd.Series([0] * len(self.rewardCol), index=self.result_table.columns, name=state))

    def insertEndRun2Table(self):
            avgReward = self.sumReward / self.numRuns
            avgScore = self.sumScore / self.numRuns
            avgSteps = self.sumSteps / self.numRuns
            
            countKey = str(self.countComplete)

            self.check_state_exist(countKey)

            self.result_table.ix[countKey, self.countIdx] = self.numRuns
            self.result_table.ix[countKey, self.rewardIdx] = avgReward
            self.result_table.ix[countKey, self.scoreIdx] = avgScore
            self.result_table.ix[countKey, self.stepsIdx] = avgSteps

            self.countComplete += 1
            self.result_table.ix[self.countCompleteKey, 0] = self.countComplete

            self.sumReward = 0
            self.sumScore = 0
            self.numRuns = 0
            self.sumSteps = 0
            print("avg results for", self.numToWrite, "trials: reward =", avgReward, "score =",  avgScore)

    def end_run(self, r, score, steps, saveTable):
        self.sumSteps += steps
        self.sumReward += r
        self.sumScore += score
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
            
        if saveTable:
            self.result_table.to_pickle(self.saveFileName, 'gzip') 

    def Reset(self):
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        
        self.check_state_exist(self.countCompleteKey)

        self.countIdx = 0
        self.rewardIdx = 1
        self.scoreIdx = 2
        self.stepsIdx = 3

        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])

        self.sumReward = 0
        self.sumScore = 0
        self.sumSteps = 0
        self.numRuns = 0
        
class ResultFile_Old:
    def __init__(self, tableName, numToWrite = 100, loadFile = True):
        self.tableName = tableName
        self.numToWrite = numToWrite

        # keys
        self.inMiddleValKey = 'middleCalculationVal'
        self.inMiddleCountKey = 'middleCalculationCount'
        self.countCompleteKey = 'countComplete'
        self.prevNumToWriteKey = 'prevNumToWrite'

        self.rewardCol = list(range(2))
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + '.gz') and loadFile:
            self.result_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        self.check_state_exist(self.prevNumToWriteKey)
        self.check_state_exist(self.inMiddleValKey)
        self.check_state_exist(self.inMiddleCountKey)
        self.check_state_exist(self.countCompleteKey)

        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])
        if numToWrite != self.result_table.ix[self.prevNumToWriteKey, 0]:
            numToWriteKey = "count_" + str(self.countComplete) + "_numToWrite"
            self.result_table.ix[numToWriteKey, 0] = numToWrite
            self.result_table.ix[numToWriteKey, 1] = 0
            self.result_table.ix[self.prevNumToWriteKey, 0] = numToWrite
        
        self.sumReward = self.result_table.ix[self.inMiddleValKey, 0]
        self.sumScore = self.result_table.ix[self.inMiddleValKey, 1]
        self.numRuns = self.result_table.ix[self.inMiddleCountKey, 0]

        if self.numRuns >= numToWrite:
            self.insertEndRun2Table()

    def check_state_exist(self, state):
        if state not in self.result_table.index:
            # append new state to q table
            self.result_table = self.result_table.append(pd.Series([0] * len(self.rewardCol), index=self.result_table.columns, name=state))

    def insertEndRun2Table(self):
            avgReward = self.sumReward / self.numRuns
            avgScore = self.sumScore / self.numRuns

            countKey = str(self.countComplete)

            self.check_state_exist(countKey)
            self.result_table.ix[countKey, 0] = avgReward
            self.result_table.ix[countKey, 1] = avgScore

            self.countComplete += 1
            self.result_table.ix[self.countCompleteKey, 0] = self.countComplete

            self.sumReward = 0
            self.sumScore = 0
            self.numRuns = 0
            print("avg results for", self.numToWrite, "trials: reward =", avgReward, "score =",  avgScore)

    def end_run(self, r, score, steps, saveTable):
        self.sumReward += r
        self.sumScore += score
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
        
        self.result_table.ix[self.inMiddleValKey, 0] = self.sumReward
        self.result_table.ix[self.inMiddleValKey, 1] = self.sumScore
        self.result_table.ix[self.inMiddleCountKey, 0] = self.numRuns
        
        if saveTable:
            self.result_table.to_pickle(self.tableName + '.gz', 'gzip') 
