import sys
import math
import pandas as pd
import numpy as np
import traceback

import importlib
import os

import matplotlib
import matplotlib.pyplot as plt

from utils import QTableParamsWOChangeInExploration
from utils import QTableParamsWithChangeInExploration

POINTS_AFTER_DOTS_2_PRINT = 4
NON_VALID_STATE_NUM = -1
def UngroupStatesAndInset2QTable(state, ungroupState, grouping, locationIdx, q_table,  currIdx):
    if currIdx == len(state):
        key = str(state)
        orgKey = str(ungroupState)
        q_table.ix[key, :] = q_table.ix[orgKey, :]
    
    elif state[currIdx] == NON_VALID_STATE_NUM:
        UngroupStatesAndInset2QTable(state, ungroupState, grouping, locationIdx, q_table, currIdx + 1)
   
    elif currIdx == locationIdx:
        fromGridSize = grouping[1][currIdx]
        toGridSize = grouping[0][currIdx]
        traverseNumOneAxis = int(grouping[0][currIdx] / grouping[1][currIdx])
        
        startX = (state[currIdx] % fromGridSize) * traverseNumOneAxis
        startY = int(state[currIdx] / fromGridSize) * traverseNumOneAxis
        for x in range (startX, startX + traverseNumOneAxis):
            for y in range (startY, startY + traverseNumOneAxis):
                state[currIdx] = x + y * toGridSize
                UngroupStatesAndInset2QTable(state, ungroupState, grouping, locationIdx, q_table, currIdx + 1)
        
        state[currIdx] = ungroupState[currIdx]

    else:
        traverseNum = int(grouping[1][currIdx] / grouping[0][currIdx])
        offset = ungroupState[currIdx]
        if grouping[0][currIdx] == 2:
            if ungroupState[currIdx] % 2 == 1:
                offset = ungroupState[currIdx] + 1
            else:
                traverseNum += 1

        for i in range(0, traverseNum):
            state[currIdx] = ungroupState[currIdx] + i * grouping[0][currIdx]
            UngroupStatesAndInset2QTable(state, ungroupState, grouping, locationIdx, q_table, currIdx + 1)

        state[currIdx] = ungroupState[currIdx]

def GetState(stateStr):
    strSplit = stateStr.split(' ') 
    state = []
    for strVal in strSplit[:]:
        if len(strVal)> 0:
            val = int(strVal)
            state.append(val)
    
    return state

def IsNewDeads(states, idx2Check, gridSize):
    deadVal = gridSize * gridSize
    for val in idx2Check[:]:
        if states[1][val] == deadVal and states[0][val] != deadVal:
            return True

    return False

def CalcChangesInTransition(t_table, stateNumVals, transitionStatesChange, includeEqual = True, locationSlots = [], gridSize = 0):
    countIdx = 0
    currValIdx = 1
    nextValIdx = 2
    equalCountIdx = 3
    deadCountIdx = stateNumVals + len(locationSlots)
    names = list(t_table.index)

    for name in names[:]:
        if name != 'TrialsData':
            name1 = name.replace("]", "")
            name1 = name1.replace("[", "")
            transitionStates = name1.split('__')

            if transitionStates[1] == 'loss' or transitionStates[1] == 'win' or transitionStates[1] == 'tie':
                continue
            states= [[], []]
            states[0] = GetState(transitionStates[0])
            states[1] = GetState(transitionStates[1])

            isNewDead = IsNewDeads(states, locationSlots, gridSize)

            actionsCount = t_table.ix[name, :]
            for a in range(0, numActions):
                numVisits = actionsCount[a]
                if numVisits > 0:
                    if isNewDead:
                        transitionStatesChange[a][deadCountIdx] += numVisits
                    locationIdx = 0
                    for i in range(0, stateNumVals):
                        idx = i + locationIdx

                        if not includeEqual and states[0][i] == states[1][i]:
                            if i in locationSlots:
                                locationIdx += 1
                                for l in range (0,2):
                                    transitionStatesChange[a][idx + l][equalCountIdx] += numVisits
                            else:
                                transitionStatesChange[a][idx][equalCountIdx] += numVisits
                        else:
                            if i in locationSlots:
                                locationIdx += 1
                                currLoc = [int(states[0][i] / gridSize), states[0][i] % gridSize]
                                nextLoc = [int(states[1][i] / gridSize), states[1][i] % gridSize]

                                for l in range (0,2):
                                    countAction = transitionStatesChange[a][idx + l][countIdx]
                                    currValAction = transitionStatesChange[a][idx + l][currValIdx]
                                    nextValAction = transitionStatesChange[a][idx + l][nextValIdx]

                                    currValAction = (currValAction * countAction + currLoc[l] * numVisits) / (countAction + numVisits)
                                    nextValAction = (nextValAction * countAction + nextLoc[l] * numVisits) / (countAction + numVisits)
                                    countAction += numVisits
                                    
                                    transitionStatesChange[a][idx + l][countIdx] = countAction
                                    transitionStatesChange[a][idx + l][currValIdx] = currValAction
                                    transitionStatesChange[a][idx + l][nextValIdx] = nextValAction

                            else:
                                countAction = transitionStatesChange[a][idx][countIdx]
                                currValAction = transitionStatesChange[a][idx][currValIdx]
                                nextValAction = transitionStatesChange[a][idx][nextValIdx]

                                currValAction = (currValAction * countAction + states[0][i] * numVisits) / (countAction + numVisits)
                                nextValAction = (nextValAction * countAction + states[1][i] * numVisits) / (countAction + numVisits)
                                countAction += numVisits
                                
                                transitionStatesChange[a][idx][countIdx] = countAction
                                transitionStatesChange[a][idx][currValIdx] = currValAction
                                transitionStatesChange[a][idx][nextValIdx] = nextValAction

    return transitionStatesChange

def PrintObjLocationCloud(locations, gridSize):

    for y in range(0, gridSize):
        for x in range(0, gridSize):
            if x + y * gridSize in locations:
                print('O', end = '')
            else:
                print('_', end = '')

        print("|")
                
def PlotExplorationChange(numTrials):
    smart = QTableParamsWithChangeInExploration()
    
    trialNum = np.arange(0, numTrials, 10)
    smartArray = np.zeros(len(trialNum), dtype=np.float, order='C')
    for i in range(0, len(trialNum)):
        smartArray[i] = smart.exploreStop + (smart.exploreStart - smart.exploreStop) * np.exp(-smart.exploreRate * trialNum[i])

    naive = QTableParamsWOChangeInExploration()
    naiveTrialNum = np.zeros(2, dtype=np.float, order='C')
    naiveArray = np.zeros(2, dtype=np.float, order='C')
    naiveArray[0] = naive.explorationProb
    naiveArray[1] = naive.explorationProb
    naiveTrialNum[1] = numTrials

    plt.plot(naiveTrialNum, naiveArray)
    plt.plot(trialNum, smartArray)

def PlotResultsFromTable(tableName, ax, newGrouping = -1):
    numSlots = 1
    table = pd.DataFrame(columns=list(range(numSlots)), dtype=np.int)
    table = pd.read_pickle(tableName + '.gz', compression='gzip')

    names = list(table.index)
    
    resultDict = {}
    oldGrouping = []
    for name in names[:]:
        if name.isdigit():
            idx = int(name)
            resultDict[idx] = table.ix[name, 0]
        elif name.find('count_') >= 0:
            start = -1
            end = -1
            for i in range(0, len(name)):
                if start == -1:
                    if name[i].isdigit():
                        start = i
                elif not name[i].isdigit():
                    end = i
                    break
            
            s = name[start:end]
            oldGrouping.append([int(s), int(table.ix[name, 0])])

    if newGrouping <= 0:
        newGrouping = oldGrouping[0][1]

    if newGrouping > 0:
        oldGrouping.append([len(resultDict), -1])
        oldGrouping.sort()
        newDict = {}

        currGroupIdx = 0
        currGrouping = oldGrouping[0][1]
        
        currCount = 0
        newVal = 0
        idx = 0
        for key, value in sorted(resultDict.items()):
            currCount += currGrouping
            newVal += value * currGrouping
            if currCount >= newGrouping:
                newDict[idx] = newVal / currCount
                idx += 1
                currCount = 0
                newVal = 0
            
            if key >= oldGrouping[currGroupIdx + 1][0]:
                currGroupIdx += 1
                currGrouping = oldGrouping[currGroupIdx][1]
        
        resultDict = newDict


    results = np.zeros(len(resultDict), dtype=np.float, order='C')
    t = np.zeros(len(resultDict), dtype=np.float, order='C')
    for key, value in sorted(resultDict.items()):
        results[key] = value
        t[key] = key * newGrouping

    ax.plot(t, results)

    return newGrouping

try:
    if len(sys.argv) < 2:
        print("Error: missing arguments")
        exit(1)
    else:
        tableType = sys.argv[1]

    if tableType == "qtable":
        if len(sys.argv) < 4:
            print("Error: missing arguments")
            exit(1)
        else:
            tableName = sys.argv[2]
            numActions = int(sys.argv[3])

        q_table = pd.DataFrame(columns=list(range(numActions)), dtype=np.float)
        q_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        f = open(tableName + '.txt', 'w')
        f.write(q_table.to_string())
        f.close()

        names = list(q_table.index)
        sumPositive = 0
        sumNegative = 0
        sumZero = 0
        numStates = 0
        
        stateNumVals = 0
        for name in names[:]:
            if name != 'TrialsData' and name != 'terminal':
                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                strSplit = name1.split(' ') 
                
                for i in range(0, len(strSplit)):
                    if len(strSplit[i])> 0:
                        stateNumVals += 1

                break

        stateValuesList = []
        for i in range (0, stateNumVals):
            stateValuesList.append([])

        forRound = math.pow(10, POINTS_AFTER_DOTS_2_PRINT) 

        for name in names[:]:
            if name == 'TrialsData':
                numRuns = q_table.ix[name, 0]
                numExpRuns = q_table.ix[name, 1]
                avgReward = q_table.ix[name, 2]
                avgExpReward = q_table.ix[name, 3]
            elif name == 'terminal':
                continue
            else:
                toPrint = False

                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                strSplit = name1.split(' ')
    
                idx = 0
                i = 0

                while idx < stateNumVals:
                    if len(strSplit[i])> 0:
                        val = int(strSplit[i])
                        if not val in stateValuesList[idx]:
                            stateValuesList[idx].append(val)
                        idx += 1              
                    i += 1

                numStates += 1
                for a in range(0, numActions):
                    val = q_table.ix[name,a]
                    if val > 0:
                        sumPositive += 1
                    elif val < 0:
                        sumNegative += 1
                    else:
                        sumZero += 1

                if toPrint:
                    print(name, end = ' ')
                    for a in range(0, numActions):
                        val = int(q_table.ix[name,a] * forRound) / forRound
                        print(val, end = ',')
                    print('')


        print("num states =" , numStates, "num positive values =", sumPositive, "num negative values =", sumNegative, "num zero values =", sumZero)
        print("num runs =", numRuns, "num experiment runs =", numExpRuns, "avg reward =", avgReward, "avg experiment reward =", avgExpReward)
        print ("num values for each state value:")
        
        i = 0
        
        for singleStateVal in stateValuesList[:]:
            print ("\tfor state[", i, "] num values = ", len(singleStateVal), "val = ", singleStateVal)
            # PrintObjLocationCloud(singleStateVal, 10)
            # print("\n\n")
            i += 1

        numAttacks = 0
        numNonAttacks = 0
        for name in names[:]:
            if name != 'TrialsData' and name != 'terminal':
                state_action = q_table.ix[name, :]
                # some actions have the same value
                state_action = state_action.reindex(np.random.permutation(state_action.index))
                action = state_action.idxmax()
                if action == 5:
                    numAttacks += 1 
                else:
                    numNonAttacks += 1
        
        print ("num attacks preffered actions =", numAttacks, "num non attacks =", numNonAttacks)
        # explore_start = 1.0
        # explore_stop = 0.01
        # decay_rate = 0.001

        # for i in range(0, 2000):
        #     exploreProb = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * i)
            # print("explore const =", exploreProb)


        # if len(sys.argv) >= 3 and sys.argv[2] == "ungroup":
        #     if len(sys.argv) >= 4:
        #         ungroupTo = list(sys.argv[3].split(','))
        #         ungroupTo = list(map(int, ungroupTo))
        #     else:
        #         ungroupTo = [1,1,1,1,1,1,UNGROUP_GRID_SIZE]

        #     grouping = []
        #     for singleStateVal in stateValuesList[:]:
        #         singleStateVal.sort()
        #         if singleStateVal[0] == -1:
        #             grouping.append(singleStateVal[2] - singleStateVal[1])
        #         else:
        #             grouping.append(singleStateVal[1] - singleStateVal[0])

        #     numValidLocs = len(stateValuesList[STATE_ENEMY_ARMY_LOCATION_IDX]) - 1
        #     grouping[STATE_ENEMY_ARMY_LOCATION_IDX] = int(math.sqrt(numValidLocs))

        #     grouping = [ungroupTo, grouping]
        #     n = 0
        #     for name in names[:]:
        #         if name != 'TrialsData':
        #             name1 = name.replace("]", "")
        #             name1 = name1.replace("[", "")
        #             strSplit = name1.split(' ')
        
        #             idx = 0
        #             i = 0

        #             state = np.zeros(stateNumVals, dtype=np.int32, order='C')
        #             ungroupState = np.zeros(stateNumVals, dtype=np.int32, order='C')

        #             while idx < stateNumVals:
        #                 if len(strSplit[i])> 0:
        #                     val = int(strSplit[i])
        #                     state[idx] = val
        #                     idx += 1              
        #                 i += 1

        #             ungroupState[:] = state[:]
        #             values = q_table.ix[name, :]
        #             UngroupStatesAndInset2QTable(state, ungroupState, grouping, STATE_ENEMY_ARMY_LOCATION_IDX, q_table, 0)
        #             print(n)
        #             n += 1

        #     print(q_table.to_string())
        #     q_table.to_pickle(Q_TABLE_TEST_NAME + '.gz', 'gzip') 

    elif tableType == "ttable":
        if len(sys.argv) < 5:
            print("Error: missing arguments")
            exit(1)

        tableNames = []

        for i in range(2, len(sys.argv)):
            if not sys.argv[i].isdigit():
                tableName = sys.argv[i]
                if os.path.isfile(tableName + '.gz'):
                    tableNames.append(tableName)
                else:
                    output = tableName
            else:
                numActions = int(sys.argv[i])

        print("tables:\n", tableNames)
        print("\nnumAction = ", numActions)
        print("output = ", output)
        print("\n")

        output_table = pd.DataFrame(columns=list(range(numActions)), dtype=np.int)

        #read tables
        columns = list(range(numActions))
        inputTables = {}
        for table in tableNames[:]:
            inputTables[table] = pd.DataFrame(columns=list(range(numActions)), dtype=np.int)
            inputTables[table] = pd.read_pickle(table + '.gz', compression='gzip')

        print("finished reading tables")
        for table in tableNames[:]:
            print("inserting table :", table)
            numCounts = 0
            names = list(inputTables[table].index)
            i = 0
            for singleName in names[:]:
                if singleName not in output_table.index:
                    output_table = output_table.append(pd.Series([0] * numActions, index=columns, name=singleName))
                
                # print("#", i, "/", len(names))
                i += 1
                for a in range(0, numActions):
                    output_table.ix[singleName, a] += inputTables[table].ix[singleName, a]
                    numCounts += inputTables[table].ix[singleName, a]

            print("for table", table, "num counts =", numCounts)


        names = list(output_table.index)  
        numCounts = 0      
        for singleName in names[:]:              
            for a in range(0, numActions):
                numCounts += output_table.ix[singleName, a]

        print("output table", table, "num counts =", numCounts)   
        
        f = open(output + '.txt', 'w')
        f.write(output_table.to_string())
        f.close()

    elif tableType == "rtable":
        if len(sys.argv) < 3:
            print("Error: missing arguments")
            exit(1)
        else:
            tableName = sys.argv[2]

        numSlots = 1
        r_table = pd.DataFrame(columns=list(range(numSlots)), dtype=np.int)
        r_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        f = open(tableName + '.txt', 'w')
        f.write(r_table.to_string())
        f.close()

    elif tableType == "results":
        if sys.argv[len(sys.argv) - 1].isdigit():
            newGrouping = int(sys.argv[len(sys.argv) - 1])
        else:
            newGrouping = -1


        tableNames = []
        for i in range(2, len(sys.argv)):
            if not sys.argv[i].isdigit():
                tableNames.append(sys.argv[i])

        
        plt.figure(1)
        PlotExplorationChange(30000)

        fig, ax = plt.subplots()  
        fileName = ""

        
        for table in tableNames[:]:
            newGrouping = PlotResultsFromTable(table, ax, newGrouping)  
            fileName += table + "_"

        fileName += ".png"
        yLabel = 'avg reward for ' + str(newGrouping) + ' trials'
        ax.set(xlabel='trials', ylabel=yLabel,
            title = 'avg reward for results tables:\n')

        ax.grid()

        fig.savefig(fileName)
        plt.legend(tableNames, loc='upper left')

        plt.show()
    else:
        print("ERROR : non valid table type!!")

except Exception as e:
        print(e)
        print(traceback.format_exc())
