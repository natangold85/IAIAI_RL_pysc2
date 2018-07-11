import sys
import math
import pandas as pd
import numpy as np
import traceback
import random
import tensorflow as tf
import os

import matplotlib
import matplotlib.pyplot as plt

import cProfile

from utils_dqn import DQN

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
            
def ResultsFromTable(tableName, grouping, dataIdx):

    table = pd.read_pickle(tableName + '.gz', compression='gzip')

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


    # table = pd.read_pickle(tableName + '.gz', compression='gzip')

    # names = list(table.index)
    
    # results = []
    # trialNum = 0
    # sumVal = 0
    # for name in names[:]:
    #     if name.isdigit():
    #         idx = int(name)
    #         subGroupSize = table.ix[name, 0]
    #         currResult = table.ix[name, dataIdx]
            
    #         trialNum += subGroupSize
    #         sumVal += subGroupSize * currResult

    #         if trialNum >= grouping:
    #             avgResult = sumVal / trialNum
    #             results.append(avgResult)
                
    #             trialNum = 0
    #             sumVal = 0

    # return np.array(results)        

try:
    if len(sys.argv) < 2:
        print("Error: missing arguments")
        exit(1)
    else:
        tableType = sys.argv[1]

    if tableType == "qtable":
        if len(sys.argv) < 3:
            print("Error: missing arguments")
            exit(1)
        else:
            tableName = sys.argv[2]

        q_table = pd.read_pickle(tableName + '.gz', compression='gzip')
        

        f = open(tableName + '.txt', 'w')
        f.write(q_table.to_string())
        f.close()

        numActions = len(q_table.columns)
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


    elif tableType == "results":
        if sys.argv[len(sys.argv) - 1].isdigit():
            grouping = int(sys.argv[len(sys.argv) - 1])
        else:
            print("\ninsert group size")
            exit()

        tableNames = []
        for i in range(2, len(sys.argv)):
            if not sys.argv[i].isdigit():
                tableNames.append(sys.argv[i])

        tableCol = ['count', 'reward', 'score', '# of steps']
        fig = plt.figure(1)

        for idx in range(1, 4):
            plt.subplot(2,2,idx)  
            for table in tableNames[:]:
                results, t = ResultsFromTable(table, grouping, idx) 
                plt.plot(t, results)
                
            plt.ylabel('avg '+ tableCol[idx] + ' for ' + str(grouping) + ' trials')
            plt.xlabel('#trials')
            plt.title('Average ' + tableCol[idx])
            plt.grid(True)
            plt.legend(tableNames, loc='best')

        fileName = "results"
        for table in tableNames[:]:
            fileName += "_" + table 
        fileName += ".png"

        fig.savefig(fileName)

        plt.show()

    elif tableType == "resultChangeFormat":
        tableName = sys.argv[2]
        grouping = int(sys.argv[3])
        table = pd.read_pickle(tableName + '.gz', compression='gzip')
        
        rewardCol = list(range(4))
        newTable = pd.DataFrame(columns=rewardCol, dtype=np.float)

        names = list(table.index)
        for name in names:
            if name.isdigit():
                newTable = newTable.append(pd.Series([0] * len(rewardCol), index=newTable.columns, name=name))
                newTable.ix[name,0] = grouping
                newTable.ix[name,1] = table.ix[name,0]
            elif name == 'countComplete':
                newTable = newTable.append(pd.Series([0] * len(rewardCol), index=newTable.columns, name=name))
                newTable.ix[name,0] = table.ix[name,0]

        newTable.to_pickle(tableName + '.gz', compression='gzip')

    elif tableType == "cmpNNToQ":
        NNFName = sys.argv[2]
        tableName = sys.argv[3]

        table = pd.read_pickle(tableName + '.gz', compression='gzip')
        numActions = len(table.columns)

        names = list(table.index)
        stateSize = 0
        for name in names[:]:
            if name != 'TrialsData' and name != 'terminal':
                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                strSplit = name1.split(' ') 
                
                for i in range(0, len(strSplit)):
                    if len(strSplit[i])> 0:
                        stateSize += 1

                break
        def build_nn_processedState(x, numActions):
            # Fully connected layers
            fc1 = tf.contrib.layers.fully_connected(x, 512)
            output = tf.nn.sigmoid(tf.contrib.layers.fully_connected(fc1, numActions)) * 2 - 1
            return output

        def ActionValuesVec(table, numActions, s):
            state_action = table.ix[s, :]
            vals = []
            for a in range(numActions):
                vals.append(state_action[a])
            return vals

        params = DQN_PARAMS(stateSize, numActions, 1, build_nn_processedState, True)
        dqn = DQN(params, NNFName, {})


        for i in range(200):
            idxState = random.randint(0, len(names) - 1)
            action = random.randint(0, numActions - 1)
            sStr = names[idxState]

            if name != 'TrialsData':
                nameStr = sStr.replace("]", "")
                nameStr = nameStr.replace("[", "")
                state = np.array(GetState(nameStr))
                vals = dqn.actionValuesVec(state)
                        
                print("NN:", vals[action], "Q:", table.ix[name, action])

    else:
        print("ERROR : non valid table type!!")

except Exception as e:
        print(e)
        print(traceback.format_exc())
