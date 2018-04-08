import sys
import math
import pandas as pd
import numpy as np
import traceback

Q_TABLE_TEST_NAME = 'q_table_test'
Q_TABLE_WITH_VESPENE_NAME = 'q_table_with_vespene'
Q_TABLE_SPARSE_NAME = 'sparse_agent_data'
Q_TABLE_BUILD_BASE_NAME = "buildbase_q_table"

T_TABLE_NAME = 'transition_table'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_TRAIN_ARMY = 'trainarmy'
ACTION_ATTACK = 'attack'

test_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_TRAIN_ARMY,
    ACTION_ATTACK,
]

vespene_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_REFINERY,
    ACTION_TRAIN_ARMY,
    ACTION_ATTACK,
]

buildbase_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_REFINERY,
    ACTION_BUILD_FACTORY,
    ACTION_TRAIN_ARMY,
]

sparse_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_TRAIN_ARMY,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            sparse_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

STATE_COMMAND_CENTER_IDX = 0
STATE_SUPPLY_DEPOT_IDX = 1
STATE_BARRACKS_IDX = 2
STATE_ARMY_IDX = 3
# STATE_MINERALS_IDX = 4
STATE_ENEMY_BASE_POWER_IDX = 4
STATE_ENEMY_ARMY_POWER_IDX = 5
STATE_ENEMY_ARMY_LOCATION_IDX = 6
STATE_NUM_VALS = 7

POINTS_AFTER_DOTS_2_PRINT = 4

NON_VALID_STATE_NUM = -1
UNGROUP_GRID_SIZE = 4

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


try:
    if len(sys.argv) < 2:
        table = "test_agent"
    else:
        table = sys.argv[1]

    if table == "test_agent":

        q_table = pd.DataFrame(columns=test_actions, dtype=np.float)
        q_table = pd.read_pickle(Q_TABLE_TEST_NAME + '.gz', compression='gzip')

        f = open(Q_TABLE_TEST_NAME + '.txt', 'w')
        f.write(q_table.to_string())
        f.close()


        names = list(q_table.index)
        sumPositive = 0
        sumNegative = 0
        sumZero = 0
        numStates = 0
        
        stateValuesList = []
        for i in range (0, STATE_NUM_VALS):
            stateValuesList.append([])

        forRound = math.pow(10, POINTS_AFTER_DOTS_2_PRINT) 

        for name in names[:]:
            if name == 'TrialsData':
                numRuns = q_table.ix[name, 0]
                numWins = q_table.ix[name, 1]
                numLoss = q_table.ix[name, 2]
                numExpWins = q_table.ix[name, 3]
                numExpLoss = q_table.ix[name, 4]
            elif name == 'terminal':
                continue
            else:
                toPrint = False

                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                strSplit = name1.split(' ')
    
                idx = 0
                i = 0

                while idx < STATE_NUM_VALS:
                    if len(strSplit[i])> 0:
                        val = int(strSplit[i])
                        if not val in stateValuesList[idx]:
                            stateValuesList[idx].append(val)
                        idx += 1              
                    i += 1

                numStates += 1
                for a in range(0, len(test_actions)):
                    val = q_table.ix[name,a]
                    if val > 0:
                        sumPositive += 1
                    elif val < 0:
                        sumNegative += 1
                    else:
                        sumZero += 1

                if toPrint:
                    print(name, end = ' ')
                    for a in range(0, len(test_actions)):
                        val = int(q_table.ix[name,a] * forRound) / forRound
                        print(val, end = ',')
                    print('')


        print("num runs =", numRuns, "num wins =", numWins, "num loss =", numLoss, "num states =", numStates)
        print("num experiment wins =", numExpWins, "num experiment loss =", numExpLoss)
        print("num positive values =", sumPositive, "num negative values =", sumNegative, "num zero values =", sumZero)
        print ("num values for each state value:")
        
        i = 0
        for singleStateVal in stateValuesList[:]:
            print ("\tfor state[", i, "] num values = ", len(singleStateVal), "val = ", singleStateVal)
            i += 1

        if len(sys.argv) >= 3 and sys.argv[2] == "ungroup":
            if len(sys.argv) >= 4:
                ungroupTo = list(sys.argv[3].split(','))
                ungroupTo = list(map(int, ungroupTo))
            else:
                ungroupTo = [1,1,1,1,1,1,UNGROUP_GRID_SIZE]

            grouping = []
            for singleStateVal in stateValuesList[:]:
                singleStateVal.sort()
                if singleStateVal[0] == -1:
                    grouping.append(singleStateVal[2] - singleStateVal[1])
                else:
                    grouping.append(singleStateVal[1] - singleStateVal[0])

            numValidLocs = len(stateValuesList[STATE_ENEMY_ARMY_LOCATION_IDX]) - 1
            grouping[STATE_ENEMY_ARMY_LOCATION_IDX] = int(math.sqrt(numValidLocs))

            grouping = [ungroupTo, grouping]
            n = 0
            for name in names[:]:
                if name != 'TrialsData':
                    name1 = name.replace("]", "")
                    name1 = name1.replace("[", "")
                    strSplit = name1.split(' ')
        
                    idx = 0
                    i = 0

                    state = np.zeros(STATE_NUM_VALS, dtype=np.int32, order='C')
                    ungroupState = np.zeros(STATE_NUM_VALS, dtype=np.int32, order='C')

                    while idx < STATE_NUM_VALS:
                        if len(strSplit[i])> 0:
                            val = int(strSplit[i])
                            state[idx] = val
                            idx += 1              
                        i += 1

                    ungroupState[:] = state[:]
                    values = q_table.ix[name, :]
                    UngroupStatesAndInset2QTable(state, ungroupState, grouping, STATE_ENEMY_ARMY_LOCATION_IDX, q_table, 0)
                    print(n)
                    n += 1

            print(q_table.to_string())
            q_table.to_pickle(Q_TABLE_TEST_NAME + '.gz', 'gzip') 


    elif table == "sparse_agent":
        q_table = pd.DataFrame(columns=sparse_actions, dtype=np.float)
        q_table = pd.read_pickle(Q_TABLE_SPARSE_NAME + '.gz', compression='gzip')

        f = open(Q_TABLE_SPARSE_NAME + '.txt', 'w')
        f.write(q_table.to_string())
        f.close()

        names = list(q_table.index)
        sumPositive = 0
        sumNegative = 0
        sumZero = 0
        numStates = 0

        for name in names[:]:
            if name == 'TrialsData':
                numRuns = q_table.ix[name, 0]
                numWins = q_table.ix[name, 1]
                numLoss = q_table.ix[name, 2]
            elif name == 'terminal':
                continue
            else:
                numStates += 1

                for a in range(0, len(sparse_actions)):
                    val = q_table.ix[name,a]
                    if val > 0:
                        sumPositive += 1
                    elif val < 0:
                        sumNegative += 1
                    else:
                        sumZero += 1


        print("num runs =", numRuns, "num wins =", numWins, "num loss =", numLoss, "num states =", numStates)
        print("num positive values =", sumPositive, "num negative values =", sumNegative, "num zero values =", sumZero)

    elif table == "vespene_agent": 
        q_table = pd.DataFrame(columns=vespene_actions, dtype=np.float)
        q_table = pd.read_pickle(Q_TABLE_WITH_VESPENE_NAME + '.gz', compression='gzip')

        f = open(Q_TABLE_WITH_VESPENE_NAME + '.txt', 'w')
        f.write(q_table.to_string())
        f.close()

        names = list(q_table.index)
        sumPositive = 0
        sumNegative = 0
        sumZero = 0
        numStates = 0
        
        stateValuesList = []
        for i in range (0, STATE_NUM_VALS):
            stateValuesList.append([])

        forRound = math.pow(10, POINTS_AFTER_DOTS_2_PRINT) 

        for name in names[:]:
            if name == 'TrialsData':
                numRuns = q_table.ix[name, 0]
                numWins = q_table.ix[name, 1]
                numLoss = q_table.ix[name, 2]
                numExpWins = q_table.ix[name, 3]
                numExpLoss = q_table.ix[name, 4]
            elif name == 'terminal':
                continue
            else:
                toPrint = False

                name1 = name.replace("]", "")
                name1 = name1.replace("[", "")
                strSplit = name1.split(' ')
    
                idx = 0
                i = 0

                while idx < STATE_NUM_VALS:
                    if len(strSplit[i])> 0:
                        val = int(strSplit[i])
                        if not val in stateValuesList[idx]:
                            stateValuesList[idx].append(val)
                        idx += 1              
                    i += 1

                numStates += 1
                for a in range(0, len(vespene_actions)):
                    val = q_table.ix[name,a]
                    if val > 0:
                        sumPositive += 1
                    elif val < 0:
                        sumNegative += 1
                    else:
                        sumZero += 1

                if toPrint:
                    print(name, end = ' ')
                    for a in range(0, len(vespene_actions)):
                        val = int(q_table.ix[name,a] * forRound) / forRound
                        print(val, end = ',')
                    print('')


        print("num runs =", numRuns, "num wins =", numWins, "num loss =", numLoss, "num states =", numStates)
        print("num experiment wins =", numExpWins, "num experiment loss =", numExpLoss)
        print("num positive values =", sumPositive, "num negative values =", sumNegative, "num zero values =", sumZero)
        print ("num values for each state value:")
        
        i = 0
        for singleStateVal in stateValuesList[:]:
            print ("\tfor state[", i, "] num values = ", len(singleStateVal), "val = ", singleStateVal)
            i += 1

    elif table == "buildbase": 
        q_table = pd.DataFrame(columns=buildbase_actions, dtype=np.float)
        q_table = pd.read_pickle(Q_TABLE_BUILD_BASE_NAME + '.gz', compression='gzip')

        f = open(Q_TABLE_BUILD_BASE_NAME + '.txt', 'w')
        f.write(q_table.to_string())
        f.close()

        names = list(q_table.index)
        sumPositive = 0
        sumNegative = 0
        sumZero = 0
        numStates = 0
        
        stateValuesList = []
        for i in range (0, STATE_NUM_VALS):
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

                while idx < STATE_NUM_VALS:
                    if len(strSplit[i])> 0:
                        val = int(strSplit[i])
                        if not val in stateValuesList[idx]:
                            stateValuesList[idx].append(val)
                        idx += 1              
                    i += 1

                numStates += 1
                for a in range(0, len(vespene_actions)):
                    val = q_table.ix[name,a]
                    if val > 0:
                        sumPositive += 1
                    elif val < 0:
                        sumNegative += 1
                    else:
                        sumZero += 1

                if toPrint:
                    print(name, end = ' ')
                    for a in range(0, len(vespene_actions)):
                        val = int(q_table.ix[name,a] * forRound) / forRound
                        print(val, end = ',')
                    print('')


        print("num runs: total =", numRuns, "experiment =", numExpRuns, "avg Reward: total =", avgReward, "experiment =", avgExpReward)
        print("num positive values =", sumPositive, "num negative values =", sumNegative, "num zero values =", sumZero)
        print ("num values for each state value:")
        
        i = 0
        for singleStateVal in stateValuesList[:]:
            print ("\tfor state[", i, "] num values = ", len(singleStateVal), "val = ", singleStateVal)
            i += 1

    elif table == "t_table":
        t_table = pd.DataFrame(columns=test_actions, dtype=np.int)
        t_table = pd.read_pickle(T_TABLE_NAME + '.gz', compression='gzip')
        
        f = open(T_TABLE_NAME + '.txt', 'w')

        


        names = list(t_table.index)
        # onlyStateTable = pd.DataFrame(columns=list(range(len(smart_actions))), dtype=np.int)
        stateDict = {}

        zeroValues = []
        for a in range(0, len(test_actions)):
            zeroValues.append(0)

        for singleName in names[:]:
            line = singleName + ' '

            strSplit = singleName.split(' ')
            notFinished = False
            idx = 0
            i = 0
            state = np.zeros(STATE_NUM_VALS, dtype=np.int32)

            while idx < STATE_NUM_VALS:
                if strSplit[i].isdigit():
                    state[idx] = int(strSplit[i])
                    idx += 1
                    
                i += 1

            key = str(state)
            if not key in stateDict.keys():
                stateDict[key] = list(zeroValues)


            for a in range(0, len(test_actions)):
                val = t_table.ix[singleName,a]

                line += str(val) + ','
                stateDict[key][a] += val

            line +='\n'
            f.write(line)
        
        for key, value in stateDict.items():
            print(key, value)

        f.close()

    elif table == "test":
        print("write test")
    else:
        print("ERROR : non valid table name!!")

except Exception as e:
        print(e)
        print(traceback.format_exc())
