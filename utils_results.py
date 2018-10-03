import math
import numpy as np
import pandas as pd
import pickle
import os.path
import time
import datetime
import sys
import threading

import tensorflow as tf

from utils_dqn import DQN

import matplotlib.pyplot as plt

from multiprocessing import Process, Lock, Value, Array, Manager

from utils import ParamsBase

def PlotResults(agentName, agentDir, runTypes, runDirectoryNames, grouping, subAgentsGroups = [""], keyResults = "results", additionPlots=[]):
    runDirectoryNames.sort()

    resultFnames = []
    groupNames = []
    fullDirectoryNames = []
    for runDirName in runDirectoryNames:
        groupNames.append(runDirName)
        
        dm_Types = eval(open("./" + runDirName + "/config.txt", "r+").read())
        runType = runTypes[dm_Types[agentName]]
        
        fName = runType[keyResults]
        
        if "directory" in runType.keys():
            dirName = runType["directory"]
        else:
            dirName = ''

        resultFnames.append(fName)
        fullDirectoryNames.append(runDirName + "/" + agentDir + dirName)
    
    print(resultFnames)
    print(fullDirectoryNames)
    plot = PlotMngr(resultFnames, fullDirectoryNames, groupNames, agentDir, subAgentsGroups)
    plot.Plot(grouping, additionPlots)

class PlotMngr:
    def __init__(self, resultFilesNamesList, resultFilesDirectories, legendList, directory2Save, subAgentsGroups):
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

            resultFile = ReadOnlyResults(name)

            self.resultFileList.append(resultFile)


        if directory2Save != '':
            if not os.path.isdir("./" + directory2Save):
                os.makedirs("./" + directory2Save)
            self.plotFName = './' + directory2Save + '/' + self.scriptName + "_resultsPlot"
        else:
            self.plotFName = self.scriptName + "_resultsPlot"

        for table in self.legendList:
            self.plotFName += "_" + table 
        
        self.plotFName += ".png"

        self.subAgentsGroups = subAgentsGroups


    def Plot(self, grouping, additionPlots):
        tableCol = ['count', 'reward', 'score', '# of steps']
        idxReward = 1

        fig = plt.figure(figsize=(19.0, 11.0))
        fig.suptitle("results for " + self.scriptName + ":", fontsize=20)

        numPlots = 1 + len(additionPlots)
        numRows = math.ceil(numPlots / 2)
        idxPlot = 1
        plt.subplot(numRows,2,idxPlot)
        
        legend = []  
        results4Addition = {}
        for iTable in range(len(self.resultFileList)):
            results, t = self.ResultsFromTable(self.resultFileList[iTable].table, grouping, idxReward) 

            switchingSubAgentsIdx = self.FindSwitchingLocations(self.resultFileList[iTable].table, t, grouping)

            for subAgentGroup in self.subAgentsGroups:
                allIdx = switchingSubAgentsIdx[subAgentGroup]
                if len(allIdx) > 0:
                    if subAgentGroup != "":
                        legend.append(self.legendList[iTable] + "_" + subAgentGroup)
                    else:
                        legend.append(self.legendList[iTable])

                    resultsTmp = np.zeros(len(results), float)
                    resultsTmp[:] = np.nan
                    resultsTmp[allIdx] = results[allIdx]

                    plt.plot(t, resultsTmp)

                    if subAgentGroup in additionPlots:
                        results4Addition[subAgentGroup] = results[allIdx]

                
            plt.ylabel('avg results for ' + str(grouping) + ' trials')
            plt.xlabel('#trials')
            plt.title('Average ' + tableCol[idxPlot])
            plt.grid(True)
            plt.legend(legend, loc='best')

            for subAgentGroup in self.subAgentsGroups:
                if subAgentGroup in results4Addition:
                    idxPlot += 1
                    plt.subplot(numRows,2,idxPlot)
                    plt.plot(results4Addition[subAgentGroup])
                    plt.ylabel('avg results for ' + str(grouping) + ' trials')
                    plt.xlabel('#trials')
                    plt.title('Average results for sub agent ' + subAgentGroup)
                    plt.grid(True)





        fig.savefig(self.plotFName)
        print("results graph saved in:", self.plotFName)

    def FindSwitchingLocations(self, table, t, grouping):
        switchingSubAgents = {}
        
        names = list(table.index)
        numRuns = np.zeros(len(names), int)
        for name in names:
            if name.isdigit():
                idx = int(name) 
                numRuns[idx] = table.ix[name, 0]   

        
        if len(self.subAgentsGroups) > 1:
            for subAgentGroup in self.subAgentsGroups:
                switchingSubAgents[subAgentGroup] = []

            for name in names:
                for subAgentGroup in self.subAgentsGroups:
                    if name.find(subAgentGroup) >= 0:
                        idxSwitch = int(table.ix[name, 0])
                        runsIdx = sum(numRuns[0:idxSwitch])
                        switchingSubAgents[subAgentGroup].append(runsIdx)
        else:
            switchingSubAgents[self.subAgentsGroups[0]] = [0]
                    

        allSwitching = []
        for key, startVals in switchingSubAgents.items():
            for val in startVals:
                allSwitching.append([val, 0, key])

        if len(allSwitching) > 0:
            allSwitching.sort()
            for idx in range(len(allSwitching) - 1):
                allSwitching[idx][1] = allSwitching[idx + 1][0]
            
            allSwitching[-1][1] = sum(numRuns)
        else:
            allSwitching.append([0, sum(numRuns), ""])
            if "" not in self.subAgentsGroups:
                self.subAgentsGroups.append("")
        
        subAgentIdx = {}
        for subAgentGroup in self.subAgentsGroups:
            subAgentIdx[subAgentGroup] = []

        for switching in allSwitching:
            start = (np.abs(t - switching[0])).argmin()
            end = (np.abs(t - (switching[1] - grouping))).argmin() + 1
            subAgentIdx[switching[2]] += list(range(start,end))

        return subAgentIdx



    def ResultsFromTable(self, table, grouping, dataIdx, groupSizeIdx = 0):
        names = list(table.index)
        tableSize = len(names) -1
        
        sumRuns = 0
        minSubGrouping = grouping
        resultsRaw = np.zeros((2, tableSize), dtype  = float)

        realSize = 0
        for name in names[:]:
            if name.isdigit():
                idx = int(name)
                subGroupSize = table.ix[name, groupSizeIdx]
                minSubGrouping = min(subGroupSize, minSubGrouping)
                resultsRaw[0, idx] = subGroupSize
                resultsRaw[1, idx] = table.ix[name, dataIdx]

                sumRuns += subGroupSize
                realSize += 1
  
        
        results = np.zeros( int(sumRuns / minSubGrouping) , dtype  = float)

        offset = 0
        for idx in range(realSize):
            
            subGroupSize = resultsRaw[0, idx]
            for i in range(int(subGroupSize / minSubGrouping)):
                results[offset] = resultsRaw[1, idx]
                offset += 1
        
        groupSizes = int(math.ceil(grouping / minSubGrouping))
        idxArray = np.arange(groupSizes)
        
        groupResults = []
        timeLine = []
        t = 0
        startIdx = groupSizes - 1
        for i in range(startIdx, len(results)):
            res = np.average(results[idxArray])
            groupResults.append(res)
            timeLine.append(t)
            idxArray += 1
            t += minSubGrouping

        return np.array(groupResults), np.array(timeLine)    

class ReadOnlyResults():
    def __init__(self, tableName):
        self.rewardCol = list(range(4))
        self.table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        print(tableName)
        if os.path.isfile(tableName + ".gz"):
            self.table = pd.read_pickle(tableName + ".gz", compression='gzip')    
        else:
            print("\n\nERROR!!")
            exit()

class ResultFile:
    def __init__(self, tableName, numToWrite = 100, loadFile = True, agentName = ''):
        self.saveFileName = tableName + '.gz'
                
        self.numToWrite = numToWrite
        self.agentName = agentName

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
            return True
        
        return False

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
            print("\t\t", threading.current_thread().getName(), ":", self.agentName, "->avg results for", self.numToWrite, "trials: reward =", avgReward, "score =",  avgScore)

    def end_run(self, r, score, steps, saveTable):
        self.sumSteps += steps
        self.sumReward += r
        self.sumScore += score
        self.numRuns += 1

        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
            
        if saveTable:
            self.result_table.to_pickle(self.saveFileName, 'gzip') 
    
    def AddSwitchSlot(self, slotName):
       
        if self.check_state_exist(slotName):
            self.result_table.ix[slotName, 0] = self.countComplete

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