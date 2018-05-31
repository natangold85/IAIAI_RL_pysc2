import sys
import math
import pandas as pd
import numpy as np
import traceback

import os

import matplotlib
import matplotlib.pyplot as plt

from utils_tables import QTableParamsWOChangeInExploration
from utils_tables import QTableParamsWithChangeInExploration

MAX_VAL = 0
POINTS_AFTER_DOTS_2_PRINT = 4
NON_VALID_STATE_NUM = -1

OLD_GRID_SIZE = 2
NEW_GRID_SIZE = 4

def GetSelectedVal(state, orgState):
    selected = orgState[len(orgState) - 1]
    if selected == 0:
        return 0
    
    idx = 0
    selectedVal = 0
    currBit = 0
    ratioToNewGridSize = int(NEW_GRID_SIZE / OLD_GRID_SIZE)
    for old in range (0, OLD_GRID_SIZE * OLD_GRID_SIZE):
        for new in range (0, ratioToNewGridSize * ratioToNewGridSize):
            if state[new + old * ratioToNewGridSize * ratioToNewGridSize] > 0:
                if selected & (1 << idx):
                    selectedVal += 1 << currBit
                currBit += 1

        if orgState[old] > 0:
            idx += 1
    return selectedVal

def GetAllPermutationStates(allPermutation, idxVec, state, currIdx = 0):
    if currIdx == len(allPermutation):
        modState = []
        for p in range(0, len(idxVec)):
            modState.append(allPermutation[p][idxVec[p]])
        return [modState] 
    
    allStates = []
    for idx in range(0, len(allPermutation[currIdx])):
        idxVec[currIdx] = idx
        modStates = GetAllPermutationStates(allPermutation, idxVec, state, currIdx + 1)
        for s in modStates:
            allStates.append(s)
    return allStates


def AllPermutationsOfLoc(val2Split, numToSplit, splitVec = [], currVal = 0, currIdx = 0):
    if numToSplit -1 == currIdx:
        v = val2Split - currVal
        vec = []
        vec[:] = splitVec[:]
        vec.append(v)
        return [vec]    
    
    perm = []
    for v in range (0, val2Split - currVal + 1):
        vec = []
        vec[:] = splitVec[:]
        vec.append(v)
        result = AllPermutationsOfLoc(val2Split, numToSplit, vec, currVal + v, currIdx + 1)
        for t in result[:]:
            perm.append(t)
  
    return perm


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


                numStates += 1
                for a in range(0, numActions):
                    val = q_table.ix[name,a]
                    if val > 0:
                        sumPositive += 1
                    elif val < 0:
                        sumNegative += 1
                    else:
                        sumZero += 1


        print("num states =" , numStates, "num positive values =", sumPositive, "num negative values =", sumNegative, "num zero values =", sumZero)
        print("num runs =", numRuns, "num experiment runs =", numExpRuns, "avg reward =", avgReward, "avg experiment reward =", avgExpReward)        

        if len(sys.argv) >= 3 and "ungroup" in sys.argv:
            outTableName = sys.argv[4]
            orgGridSize = 2
            newGridSize = 2
            scaleRatio = newGridSize / orgGridSize
            bucketing = 4
            numActions = list(range(2 + newGridSize * newGridSize * 2))
            outTable = pd.DataFrame(columns = numActions, dtype=np.float)

            names = list(q_table.index)
            toPrint = []
            for i in range(0,100):
                toPrint.append(i)

            allSize = len(names)
            currIdx = 0

            for name in names[:]:
                currIdx += 1
                pct = ((currIdx * 100) / allSize)
                print(name)
                if pct in toPrint:
                    print("passed", pct, "%")
                if name != 'TrialsData':
                    nameStr = name.replace("]", "")
                    nameStr = nameStr.replace("[", "")
                    state = GetState(nameStr)

                    offset = 0
                    allPermutation = []

                    for loc in range(0, 2 * orgGridSize * orgGridSize):
                        powerGroup = state[loc]
                        perm = []
                        if powerGroup > 0:
                            start = powerGroup - bucketing + 1
                            end = powerGroup + 1
                            for power in range(start, end):
                                perm.append(power)
                        else:
                            perm.append(powerGroup)
                        
                        allPermutation.append(perm)

                    allPermutation.append([state[offset + 1]])
                    allPermutation.append([state[offset + 2]])




                    # for side in range(0, 2):
                    #     for loc in range(0, orgGridSize * orgGridSize):
                    #         powerGroup = state[offset]
                    #         if powerGroup > 0:
                    #             start = (powerGroup - 1) * bucketing + 1
                    #             end = powerGroup * bucketing + 1
                    #             for power in range(start, end):
                    #                 perm = AllPermutationsOfLoc(power, scaleRatio * scaleRatio)
                    #         else:
                    #             perm = [[]]
                    #             for i in range(0, int(scaleRatio * scaleRatio)):
                    #                 perm[0].append(0)

                    #         allPermutation.append(perm)
                    #         offset += 1
                    
                    idxVec = []
                    for i in range(0, len(allPermutation)):
                        idxVec.append(0)
                                    
                    allStates = GetAllPermutationStates(allPermutation, idxVec, state)
                    for s in allStates:
                        outTable = outTable.append(pd.Series([0] * len(numActions), index=outTable.columns, name=str(s)))
                        for a in range (0, len(numActions)):
                            val = q_table.ix[name, a]
                            outTable.ix[str(s), a] = val
                else:
                    outTable = outTable.append(pd.Series([0] * len(numActions), index=outTable.columns, name=name))
                    for a in range (0, len(numActions)):
                        val = q_table.ix[name, a]
                        outTable.ix[name, a] = val

            print(outTable.to_string())
            outTable.to_pickle(outTableName + '.gz', 'gzip') 

    elif tableType == "ttable":
        if len(sys.argv) < 4:
            print("Error: missing arguments")
            exit(1)
        else:
            tableName = sys.argv[2]
            numActions = int(sys.argv[3])

        t_table = pd.DataFrame(columns=list(range(numActions)), dtype=np.float)
        t_table = pd.read_pickle(tableName + '.gz', compression='gzip')

        f = open(tableName + '.txt', 'w')

        names = list(t_table.index)
        numWins = 0
        numLoss = 0

        for name in names[:]:
            if name != "TrialsData":
                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                states = name1.split("__")
                s = GetState(states[0])
                numStateVals = len(s)
                break
    
        actionsTransition = []
        for a in range (0, numActions):
            actionT = []
            for stateVal in range(0, numStateVals):
                actionT.append([0,0,0])
            actionsTransition.append(actionT) 

        locationChange = 0
        locationStatic = 0
        nonExpectedChange = 0
        for name in names[:]:
            line = name
            
            insert2Calc = False
            if name != "TrialsData":
                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                states = name1.split("__")

                if states[1] == "win":
                    numWins += sum(t_table.ix[name, :])
                elif states[1] == "loss":
                    numLoss += sum(t_table.ix[name, :])
                else:
                    s = GetState(states[0])
                    s_ = GetState(states[1])
                    insert2Calc = True

            for a in range (0, numActions):
                val = t_table.ix[name, a]
                line += "   " + str(val)
                if insert2Calc and val > 0:
                    for stateVal in range(0, numStateVals):
                        actionsTransition[a][stateVal][0] += s[stateVal] * val
                        actionsTransition[a][stateVal][1] += s_[stateVal] * val
                        actionsTransition[a][stateVal][2] += val
                    
                    if a > 4*4 + 1:
                        goTo = a - 4*4 + 1
                        selectedVal = s[2*4*4]
                        if selectedVal > 0:
                            idx = 0
                            for i in range(0, 4*4):
                                if s[i] > 0:
                                    if 1 << idx & selectedVal > 0:
                                        if goTo != i:
                                            if s_[i] != s[i]:
                                                locationChange += 1
                                            else:
                                                locationStatic += 1
                                    idx += 1
                        else:
                            for i in range(0, 4*4):
                                if s[i] < s_[i]:
                                    nonExpectedChange += 1
                                        
                    else:
                        for i in range(0, 4*4):
                            if s[i] < s_[i]:
                                nonExpectedChange += 1

            line += "\n"

            f.write(line)
        f.close()

        print("after valid move action change =", locationChange, "static =", locationStatic)
        print("non expected change =", nonExpectedChange)

        table = t_table[t_table.index != "TrialsData"]
        print(table.describe())
        # for a in range (0, numActions):
        #     print("for action = ", a, ":\n")
        #     for stateVal in range(0, numStateVals):
        #         print(actionsTransition[a][stateVal][0] / actionsTransition[a][stateVal][2], actionsTransition[a][stateVal][1] / actionsTransition[a][stateVal][2])

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

        
        # plt.figure(1)
        # PlotExplorationChange(30000)

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

    elif tableType == "paste_results":
        tableName = sys.argv[2]
        outTableName = sys.argv[4]
        part2Include = int(sys.argv[3])
        numSlots = 1
        table = pd.DataFrame(columns=list(range(numSlots)), dtype=np.int)
        table = pd.read_pickle(tableName + '.gz', compression='gzip')

        num2Write = table.ix['prevNumToWrite',0]
        end =  int(table.ix['countComplete',0])
        print("before:")
        print(table.to_string())

    
        start2Drop = int(math.floor(part2Include / num2Write))
        
        leftOver = int(part2Include % num2Write)
        table.ix['middleCalculationCount',0] = leftOver
        table.ix['middleCalculationVal',0] = table.ix[str(start2Drop),0]

        namesToDrop = []
        for toDrop in range(start2Drop, end):
            namesToDrop.append(str(toDrop))
            table = table[table.index != str(toDrop)]
        
        table.ix['countComplete',0] = start2Drop

        print("\n\n\n\nafter:")
        print(table.to_string())

        table.to_pickle(outTableName + '.gz', 'gzip') 
    else:
        print("ERROR : non valid table type!!")

except Exception as e:
        print(e)
        print(traceback.format_exc())
