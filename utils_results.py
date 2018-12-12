import math
import numpy as np
import pandas as pd
import pickle
import os.path
import time
import datetime
import sys
import threading
import os

import tensorflow as tf

from algo_dqn import DQN

import matplotlib.pyplot as plt

from multiprocessing import Process, Lock, Value, Array, Manager

from utils_plot import PlotMeanWithInterval

from agentRunTypes import GetRunType

def AvgResults(path, name, idxDir=None, key4Results=None):
    # create results read only class according to args
    if idxDir == None:
        results = ReadOnlyResults(path + '/' + name)
    else:
        results = ReadOnlyResults(path + '_' + str(idxDir) + '/' + name)

    # return avg reward
    return results.AvgReward(key4Results)

def ChangeName2NextResultFile(path, name, idxDir, idx2CurrentFile):
    # 
    srcFName = path + '_' + str(idxDir) + '/' + name + ".gz"
    newFName = path + '_' + str(idxDir) + '/' + name + "_" + str(idx2CurrentFile) +".gz"
    
    # if results exist reanme file name to newFName
    if os.path.isfile(srcFName):
        if os.path.isfile(newFName):
            os.remove(newFName)
        os.rename(srcFName, newFName)
        return True

    return False

def PlotResults(agentName, runDirectoryNames, grouping, subAgentsGroups = [""], keyResults = "results", additionPlots=[], maxTrials2Plot=None, multipleDm=False):
    # sort for fixed location of plots
    runDirectoryNames.sort()

    resultFnames = []
    groupNames = []
    fullDirectoryNames = []
    
    # run on each directory
    for runDirName in runDirectoryNames:
        groupNames.append(runDirName)
        
        # read config and get run type (for paths)
        configDict = eval(open("./" + runDirName + "/config.txt", "r+").read())
        runType = GetRunType(agentName, configDict)
        
        fName = runType[keyResults]
        
        if "directory" in runType.keys():
            dirName = runType["directory"]
        else:
            dirName = ''

        resultFnames.append(fName)
        fullDirectoryNames.append(runDirName + "/" + agentName + "/" + dirName)
    
    print(fullDirectoryNames)
    # plot results
    plot = PlotMngr(resultFnames, fullDirectoryNames, groupNames, agentName, subAgentsGroups, multipleDm)
    plot.Plot(grouping, additionPlots, maxTrials2Plot, multipleDm)

class PlotMngr:
    def __init__(self, resultFilesNamesList, resultFilesDirectories, groupNames, directory2Save, subAgentsGroups, multipleDm):
        # list of all necessary ReadOnlyResults class (in case of multiple decision maker the list is 2 dimensions)
        self.resultFileList = []

        # sub groups for 1 learning
        self.subAgentsGroups = subAgentsGroups

        # set file name for plot
        self.scriptName = sys.argv[0]
        self.scriptName = self.scriptName.replace(".\\", "")
        self.scriptName = self.scriptName.replace(".py", "")

        if directory2Save != '':
            if not os.path.isdir("./" + directory2Save):
                os.makedirs("./" + directory2Save)
            self.plotFName = './' + directory2Save + '/' + self.scriptName + "_resultsPlot"
        else:
            self.plotFName = self.scriptName + "_resultsPlot"

        for group in groupNames:
            self.plotFName += "_" + group 
        
        self.plotFName += ".png"

        # read results from files

        self.legendList = []
        for i in range(len(resultFilesNamesList)):
            # read from each group all instances
            name = './' + resultFilesDirectories[i]
            resultsInstances = []
            instanceName = []
            dmInstance = 0
            currName = name + "_" + str(dmInstance)

            if os.path.isdir(name):
                resultFile = ReadOnlyResults(name + '/' + resultFilesNamesList[i])
                resultsInstances.append(resultFile)
                instanceName.append("")
            
            while os.path.isdir(currName):
                resultFile = ReadOnlyResults(currName + '/' + resultFilesNamesList[i])
                resultsInstances.append(resultFile)
                dmInstance += 1
                currName = name + "_" + str(dmInstance)
                instanceName.append("_" + str(dmInstance))
            
            # insert to list Results instances
            if multipleDm:
                self.resultFileList.append(resultsInstances)
                self.legendList += groupNames[i]
            else:
                self.resultFileList += resultsInstances
                legendResultGroup = [groupNames[i] + ins for ins in instanceName]
                self.legendList += legendResultGroup



    def Plot(self, grouping, additionPlots, maxTrials2Plots, multipleDm):
        idxReward = 1

        fig = plt.figure(figsize=(19.0, 11.0))
        fig.suptitle("results for " + self.scriptName + ":", fontsize=20)

        numPlots = 1 + len(additionPlots)
        numRows = math.ceil(numPlots / 2)
        idxPlot = 1
        plt.subplot(numRows,2,idxPlot)
        
        legend = []  
        results4Addition = {}
        for idxResults in range(len(self.resultFileList)):
            # if multiple groups needs to extract avg and std from results (switching sub agent idx is learning switch in the same run)
            if multipleDm:
                results, t, switchingSubAgentsIdx, stdResults = self.AvgResultsFromMultipleTables(idxResults, grouping, idxReward, maxTrials2Plots)
            else:
                results, t = self.ResultsFromTable(self.resultFileList[idxResults].table, grouping, idxReward, maxTrials2Plots) 
                switchingSubAgentsIdx = self.FindSwitchingLocations(self.resultFileList[idxResults].table, t, grouping)

            # in case there is no learning switching this for will run 1 time
            for subAgentGroup in self.subAgentsGroups:
                groupIdx = switchingSubAgentsIdx[subAgentGroup]
                if len(groupIdx) > 0:
                    # append sub group names to legend
                    if subAgentGroup != "":
                        legend.append(self.legendList[idxResults] + "_" + subAgentGroup)
                    else:
                        legend.append(self.legendList[idxResults])

                    resultsTmp = np.zeros(len(results), float)
                    resultsTmp[:] = np.nan
                    resultsTmp[groupIdx] = results[groupIdx]

                    # plot results
                    if multipleDm:
                        stdTmp = np.zeros(len(stdResults), float)
                        stdTmp[:] = np.nan
                        stdTmp[groupIdx] = stdResults[groupIdx] 
                        PlotMeanWithInterval(t, resultsTmp, stdTmp)
                    else:
                        plt.plot(t, resultsTmp)

                    # if needed to plot sub group individual
                    if subAgentGroup in additionPlots:
                        results4Addition[subAgentGroup] = results[groupIdx]

                
            plt.ylabel('avg reward for ' + str(grouping) + ' trials')
            plt.xlabel('#trials')
            plt.title('Average Reward')
            plt.grid(True)
            plt.legend(legend, loc='best')

            for subAgentGroup in self.subAgentsGroups:
                if subAgentGroup in results4Addition:
                    idxPlot += 1
                    plt.subplot(numRows,2,idxPlot)
                    plt.plot(results4Addition[subAgentGroup])
                    plt.ylabel('avg reward for ' + str(grouping) + ' trials')
                    plt.xlabel('#trials')
                    plt.title('Average results for sub agent ' + subAgentGroup)
                    plt.grid(True)

        # save figure
        fig.savefig(self.plotFName)
        print("results graph saved in:", self.plotFName)

    def AvgResultsFromMultipleTables(self, idxResults, grouping, idxReward, maxTrials2Plots):
        allResults = []
        allT = []
        maxLen = 0
        maxLenIdx = -1
        tableIdx = 0

        for singleTable in self.resultFileList[idxResults]:
            results, t = self.ResultsFromTable(singleTable.table, grouping, idxReward, maxTrials2Plots) 
            # if results exist append them
            if len(results) > 0:
                allResults.append(results)
                allT.append(t)
                # calculate max len of results
                if len(t) > maxLen:
                    maxLen = len(t)
                    maxLenIdx = len(allT) - 1

            tableIdx += 1
        
        # fill with nan to compare lengths
        for i in range(len(allResults)):
            newResults = np.zeros(maxLen)
            newResults.fill(np.nan)
            newResults[:len(allResults[i])] = allResults[i]
            allResults[i] = newResults
            allT[i] = allT[maxLenIdx]

        # find learning switching
        switchingSubAgentsIdx = self.FindSwitchingLocations(self.resultFileList[idxResults][maxLenIdx].table, allT[maxLenIdx], grouping)
    
        return np.nanmean(allResults, axis=0), allT[maxLenIdx], switchingSubAgentsIdx, np.nanstd(allResults, axis=0)
        
    def FindSwitchingLocations(self, table, t, grouping):
        switchingSubAgents = {}
        
        names = list(table.index)

        numRuns = np.zeros(len(names), int)
        for name in names:
            if name.isdigit():
                idx = int(name) 
                numRuns[idx] = table.ix[name, 0]   

        # if supposed to be switching calcullate switching locations
        if len(self.subAgentsGroups) > 1:
            for subAgentGroup in self.subAgentsGroups:
                switchingSubAgents[subAgentGroup] = []

            for name in names:
                for subAgentGroup in self.subAgentsGroups:
                    # find switching location of specific group
                    if name.find(subAgentGroup) >= 0:
                        idxSwitch = int(table.ix[name, 0])
                        runsIdx = sum(numRuns[0:idxSwitch])
                        switchingSubAgents[subAgentGroup].append(runsIdx)
        else:
            switchingSubAgents[self.subAgentsGroups[0]] = [0]
                    
        # for all switching create list [startpoint, endPoint, name]
        allSwitching = []
        for key, startVals in switchingSubAgents.items():
            for val in startVals:
                allSwitching.append([val, 0, key])

        if len(allSwitching) > 0:
            # sort switching according to start point
            allSwitching.sort()
            # insert end point to data(start point of next)
            for idx in range(len(allSwitching) - 1):
                allSwitching[idx][1] = allSwitching[idx + 1][0]
            
            # end point of the last switching
            allSwitching[-1][1] = sum(numRuns)
        
        else:
            allSwitching.append([0, sum(numRuns), ""])
            if "" not in self.subAgentsGroups:
                self.subAgentsGroups.append("")
        
        subAgentIdx = {}
        for subAgentGroup in self.subAgentsGroups:
            subAgentIdx[subAgentGroup] = []

        # find idx of switching according to t
        for switching in allSwitching:
            start = (np.abs(t - switching[0])).argmin()
            end = (np.abs(t - (switching[1] - grouping))).argmin() + 1
            subAgentIdx[switching[2]] += list(range(start,end))

        return subAgentIdx



    def ResultsFromTable(self, table, grouping, dataIdx, maxTrials2Plot, groupSizeIdx = 0):
        names = list(table.index)
        tableSize = len(names) -1
        
        resultsRaw = np.zeros((2, tableSize), dtype  = float)
        sumRuns = 0
        realSize = 0

        # min grouping in table in case its changed during run
        minSubGrouping = grouping
        
        for name in names[:]:
            if name.isdigit():
                idx = int(name)
                subGroupSize = table.ix[name, groupSizeIdx]
                minSubGrouping = min(subGroupSize, minSubGrouping)
                resultsRaw[0, idx] = subGroupSize
                resultsRaw[1, idx] = table.ix[name, dataIdx]

                sumRuns += subGroupSize
                realSize += 1
                if maxTrials2Plot != None and maxTrials2Plot < sumRuns:
                    break
  
        # create results for equal grouping
        results = np.zeros( int(sumRuns / minSubGrouping) , dtype  = float)

        offset = 0
        for idx in range(realSize):
            
            subGroupSize = resultsRaw[0, idx]
            for i in range(int(subGroupSize / minSubGrouping)):
                results[offset] = resultsRaw[1, idx]
                offset += 1
        
        # transfer results to requested grouping
        
        groupSizes = int(math.ceil(grouping / minSubGrouping))
        # for calculation of average of group
        idxArray = np.arange(groupSizes)
        
        groupResults = []
        timeLine = []
        t = 0
        startIdx = groupSizes - 1
        for i in range(startIdx, len(results)):
            groupResults.append(np.average(results[idxArray]))
            timeLine.append(t)
            idxArray += 1
            t += minSubGrouping

        return np.array(groupResults), np.array(timeLine)    

class ReadOnlyResults():
    def __init__(self, tableName):
        self.rewardCol = list(range(4))
        
        self.countIdx = 0
        self.rewardIdx = 1
        self.scoreIdx = 2
        self.stepsIdx = 3

        self.tableName = tableName
        self.table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        if os.path.isfile(tableName + ".gz") and os.path.getsize(tableName + '.gz') > 0:
            self.table = pd.read_pickle(tableName + ".gz", compression='gzip')

            
    
    def AvgReward(self, key = None):
        names = list(self.table.index)
        if len(names) == 0:
            return None
        
        # calulate sum reward and count
        sumVal = 0.0
        count = 0
        for i in range(len(names)):
            k = str(i)
            if k in self.table.index:
                v = self.table.ix[k, self.rewardIdx]
                c = self.table.ix[k, self.countIdx]
                sumVal += v * c
                count += c
        
        return sumVal / count if count > 0 else None

            


class ResultFile:
    def __init__(self, tableName, numToWrite = 100, agentName = ''):
        self.saveFileName = tableName
                
        self.numToWrite = numToWrite
        self.agentName = agentName

        self.rewardCol = list(range(4))
        self.countCompleteKey = 'countComplete'
        self.countIdx = 0
        self.rewardIdx = 1
        self.scoreIdx = 2
        self.stepsIdx = 3

        self.sumReward = 0
        self.sumScore = 0
        self.sumSteps = 0
        self.numRuns = 0

        self.InitTable()

    def InitTable(self):
        # init table and create complete count key
        self.result_table = pd.DataFrame(columns=self.rewardCol, dtype=np.float)
        self.check_state_exist(self.countCompleteKey)
        self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])

    def Load(self):
        # if file exist reaf table and read from table complete count key
        if os.path.isfile(self.saveFileName + '.gz') and os.path.getsize(self.saveFileName + '.gz') > 0:
            self.result_table = pd.read_pickle(self.saveFileName + '.gz', compression='gzip')
            self.countComplete = int(self.result_table.ix[self.countCompleteKey, 0])

    def Save(self):
        self.result_table.to_pickle(self.saveFileName + '.gz', 'gzip') 
    
    def NumRuns(self):
        # if num to write changes during run this function return wrong results
        return self.countComplete * self.numToWrite + self.numRuns

    def check_state_exist(self, state):
        if state not in self.result_table.index:
            # append new state filled with 0
            self.result_table = self.result_table.append(pd.Series([0] * len(self.rewardCol), index=self.result_table.columns, name=state))
            return True
        
        return False

    def insertEndRun2Table(self):
        # calculate results
        avgReward = self.sumReward / self.numRuns
        avgScore = self.sumScore / self.numRuns
        avgSteps = self.sumSteps / self.numRuns
        
        # find key and create state
        countKey = str(self.countComplete)
        self.check_state_exist(countKey)

        # insert values to table
        self.result_table.ix[countKey, self.countIdx] = self.numRuns
        self.result_table.ix[countKey, self.rewardIdx] = avgReward
        self.result_table.ix[countKey, self.scoreIdx] = avgScore
        self.result_table.ix[countKey, self.stepsIdx] = avgSteps

        # update count complete key
        self.countComplete += 1
        self.result_table.ix[self.countCompleteKey, 0] = self.countComplete

        # reset current results
        self.sumReward = 0
        self.sumScore = 0
        self.numRuns = 0
        self.sumSteps = 0

        print("\t\t", threading.current_thread().getName(), ":", self.agentName, "->avg results for", self.numToWrite, "trials: reward =", avgReward, "score =", avgScore, "numRun =", self.NumRuns())

    def end_run(self, r, score, steps, saveTable):
        # insert results
        self.sumSteps += steps
        self.sumReward += r
        self.sumScore += score
        self.numRuns += 1

        # save curr results in table if necessary
        if self.numRuns == self.numToWrite:
            self.insertEndRun2Table()
        
        # save table if needed
        if saveTable:
            self.Save()

    def AddSlot(self, slotName):
        # add slot for switching location
        if self.check_state_exist(slotName):
            self.result_table.ix[slotName, 0] = self.countComplete

    def Reset(self):
        self.InitTable()

        # reset result that not saved to table
        self.sumReward = 0
        self.sumScore = 0
        self.sumSteps = 0
        self.numRuns = 0