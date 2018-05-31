import time
import datetime
import random
import numpy as np
import pandas as pd
import math
import sys

import matplotlib
import matplotlib.pyplot as plt

TTABLE_IDX_ACTIONS_COUNT = 1
TTABLE_IDX_TABLE = 0

class TreeData:
    def __init__(self):
        # solver params
        self.LEARNING_RATE = 0.001
        
        self.DISCOUNT = 0.95
        
        self.EXPLORE_STOP = 0.01
        self.EXPLORE_START = 1.0
        self.EXPLORE_RATE = 0.0001

        self.SIZE_TRANSITION_TO_COMPLETE_TABLE = 5
        # tables
        self.MIN_TRANSITIONS_TO_CONTINUE_SIM = 2
        self.TRANSITION_TABLE = None
        self.Q_TABLE = None

        # model params
        self.NUM_ACTIONS = None
        self.TERMINAL_STATES = ['win', 'loss', 'tie', 'terminal']
        
        self.REWARDS = {}
        self.REWARDS['win'] = 1
        self.REWARDS['loss'] = -1
        self.REWARDS['tie'] = 0
        self.REWARDS['terminal'] = 0

        self.MIN_REWARD = min(self.REWARDS.values())

    def ExplorationFactor(self, count):
        return self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP) * np.exp(-self.EXPLORE_RATE * count)

TREE_DATA = TreeData()
test = 0 
class TreeMngr:
    def __init__(self, transitionTable, qtable, numActions, searchDuration):
        
        TREE_DATA.NUM_ACTIONS = numActions
        TREE_DATA.Q_TABLE = qtable
        TREE_DATA.TRANSITION_TABLE = transitionTable

        self.searchDuration = searchDuration
        self.currRoot = None

    def Hallucinate(self, state, prevAction):
        if self.currRoot != None and prevAction != None:
            self.TrimRoot(state, prevAction)
        else:
            self.currRoot = StateNode(state)


        startTime = datetime.datetime.now()
        currTime = datetime.datetime.now()

        while (currTime - startTime).total_seconds() < self.searchDuration:
            self.currRoot.Simulate()
            currTime = datetime.datetime.now()


    def TrimRoot(self, state, prevAction):
        actionNode = self.currRoot.actionChilds[prevAction]

        if state in actionNode.stateChilds:
            self.currRoot = actionNode.stateChilds[state]
        else:
            self.currRoot = StateNode(state)

    def ChooseAction(self):
        if self.currRoot == None:
            return None
        
        return self.currRoot.ChooseAction()

    def Value(self,action):
        if self.currRoot == None:
            return None
        
        return self.currRoot.Value(action)
        
    def ExecuteAction(self, state, action):
        if state in TREE_DATA.TRANSITION_TABLE:
            return self.currRoot.FindNextState(state, action)
        else:
            return "terminal"

    def UpdateQTable(self):
        if self.currRoot != None:
            self.currRoot.UpdateQTable()

    def Depth(self):
        if self.currRoot != None:
            return self.currRoot.Depth() + 1
        else:
            return 0
    
    def Size(self):
        if self.currRoot != None:
            return self.currRoot.Size()
        else:
            return 0

  
class StateNode:
    def __init__(self, state, count = 0):   
        self.state = state 
        self.count = count 
        self.actionChilds = []
    
    def Simulate(self, currIdx = 0):
        if len(self.actionChilds) == 0:
            self.CreateActionChilds()

        action = self.UpperBoundAction()
        # print("action chosen =", action)

        nextState = self.FindNextState(self.state, action)

        # print("real depth =", currIdx, "simulate depth =", self.Depth(), "state =", self.state, "\n\tnext state =", nextState)
        if nextState in TREE_DATA.TERMINAL_STATES:
            reward = TREE_DATA.REWARDS[nextState]
        elif nextState not in self.actionChilds[action].stateChilds:
            reward = TREE_DATA.DISCOUNT * self.Rollout(nextState)
            self.actionChilds[action].AddState(nextState)
        else:
            reward = TREE_DATA.DISCOUNT * self.actionChilds[action].stateChilds[nextState].Simulate(currIdx + 1)

        self.UpdateReward(action, reward)
        
        return reward
        
    def Rollout(self, state, currIdx = 0):
        if state in TREE_DATA.Q_TABLE:
            action = TREE_DATA.Q_TABLE.ix[state, :].idxmax()
        else:
            action = random.randint(0, TREE_DATA.NUM_ACTIONS - 1)

        nextState = self.FindNextState(state, action)
        # print(state, " -> ", nextState)
        if nextState in TREE_DATA.TERMINAL_STATES:
            # print("rollout terminal")
            return TREE_DATA.REWARDS[nextState]
        else:
            return TREE_DATA.DISCOUNT * self.Rollout(nextState, currIdx + 1)


    def CreateActionChilds(self):
        if self.state in TREE_DATA.Q_TABLE.index:
            valueVector = TREE_DATA.Q_TABLE.ix[self.state, :]
        else:           
            valueVector = np.zeros(TREE_DATA.NUM_ACTIONS, dtype=np.float, order='C')

        for a in range(0, TREE_DATA.NUM_ACTIONS):
            self.actionChilds.append(ActionNode(self.state, a, valueVector[a]))
            
        

    def UpperBoundAction(self):
        bestValue = TREE_DATA.MIN_REWARD - 1
        bestAction = -1
        for a in range(0, TREE_DATA.NUM_ACTIONS):
            if self.actionChilds[a].count == 0:
                return a
            
            val = self.actionChilds[a].value + TREE_DATA.ExplorationFactor(self.count) * math.sqrt(math.log(self.count + 1) / self.actionChilds[a].count)
            if val > bestValue:
                bestValue = val
                bestAction = a
        
        return bestAction

    def ChooseAction(self):
        if len(self.actionChilds) == 0:
            return None
        
        idxMax = -1
        rewardMax = TREE_DATA.MIN_REWARD - 1

        for a in range (0, TREE_DATA.NUM_ACTIONS):
            if self.actionChilds[a].value > rewardMax:
                idxMax = a
                rewardMax = self.actionChilds[a].value

        return idxMax

    def Value(self, action):
        if len(self.actionChilds) == 0:
            return None
        
        return self.actionChilds[action].value
        
    def UpdateQTable(self):
        for a in self.actionChilds:
            a.UpdateQTable()


    def FindNextState(self, state, action):
        stateSize = TREE_DATA.TRANSITION_TABLE[state][TTABLE_IDX_ACTIONS_COUNT][action]
        if stateSize < TREE_DATA.MIN_TRANSITIONS_TO_CONTINUE_SIM:
            return "terminal"

        stateNum = random.randint(0, stateSize)

        for s_ in TREE_DATA.TRANSITION_TABLE[state][TTABLE_IDX_TABLE].index:
            stateNum -= TREE_DATA.TRANSITION_TABLE[state][TTABLE_IDX_TABLE].ix[s_, action]
            if stateNum <= 0:
                return s_

    def UpdateReward(self, action, reward):
        self.actionChilds[action].UpdateReward(reward)
        self.count += 1

    def Depth(self):
        d = 0
        for a in self.actionChilds:
            d = max(d, a.Depth())
        
        return d + 1

    def Size(self):
        size = 0
        for a in self.actionChilds:
            size += a.Size()
        
        return size       

class ActionNode:
    def __init__(self, state, action, value = 0):
        self.fatherState = state
        self.action = action
        self.stateChilds = {}
        self.value = value
        self.count = 10
    
    def AddState(self, state, count = 0):
        self.stateChilds[state] = StateNode(state, count)
            
    def UpdateReward(self, reward):
        self.value = (self.value * self.count + reward) / (self.count + 1)
        self.count += 1

    def Depth(self):
        d = 0
        for state in self.stateChilds.values():
            d = max(d, state.Depth())

        return d

    def UpdateQTable(self):    
        if TREE_DATA.TRANSITION_TABLE[self.fatherState][TTABLE_IDX_ACTIONS_COUNT][self.action] >= TREE_DATA.SIZE_TRANSITION_TO_COMPLETE_TABLE:
            predict = TREE_DATA.Q_TABLE.ix[self.fatherState, self.action]
            TREE_DATA.Q_TABLE.ix[self.fatherState, self.action] = predict + TREE_DATA.LEARNING_RATE * (self.value - predict)   

        # for s in self.stateChilds.values():
        #     if s.count > 0:
        #         s.UpdateQTable()

    def Size(self):
        size = len(self.stateChilds)
        for state in self.stateChilds.values():
            size += state.Size()
        
        return size   

if len(sys.argv) < 3:
    print("enter table names")

qTableName = sys.argv[1]
tTableName = sys.argv[2]
numActions = int(sys.argv[3])


qtable = pd.read_pickle(qTableName + '.gz', compression='gzip')

#print(qtable[qtable.index != "TrialsData"].describe())
ttable = pd.read_pickle(tTableName + '.gz', compression='gzip')
treeMngr = TreeMngr(ttable, qtable, numActions, 0.2)

action = None
bestValueHallucinator = []
countValueHallucinator = []
hallucinatorResults = 0

bestValueTable = []
countValueTable = []
tableResults = 0

maxNumSteps = 100
numTrials2Save = 10
for i in range(0, maxNumSteps):
    bestValueHallucinator.append(0)
    countValueHallucinator.append(0)
    bestValueTable.append(0)
    countValueTable.append(0)

startTime = datetime.datetime.now()
for t in range(0,100000):

    numVals = 0
    while numVals < 50:
        state = random.choice(list(ttable.keys()))
        if type(ttable[state])==list and len(ttable[state]) > 1:
            numVals = sum(ttable[state][1])
    stateOnlyTable = ''.join(state)

    terminalTable = False
    terminalHallucinator = False
    for i in range(0, maxNumSteps):
        if not terminalHallucinator:
            treeMngr.Hallucinate(state, action)
            action = treeMngr.ChooseAction()
            value = treeMngr.Value(action)
            treeMngr.UpdateQTable()

            bestValueHallucinator[i] += value
            countValueHallucinator[i] += 1
            
            state = treeMngr.ExecuteAction(state, action)
            if state in TREE_DATA.TERMINAL_STATES:
                terminalHallucinator = True         

        if not terminalTable:
            onlyTableAction = TREE_DATA.Q_TABLE.ix[stateOnlyTable, :].idxmax()
            value = qtable.ix[stateOnlyTable,onlyTableAction]

            bestValueTable[i] += value
            countValueTable[i] += 1

            stateOnlyTable = treeMngr.ExecuteAction(stateOnlyTable, action)
            if stateOnlyTable in TREE_DATA.TERMINAL_STATES:
                terminalTable = True          
         
        if terminalTable and terminalHallucinator:
            break
    
    if state in TREE_DATA.TERMINAL_STATES:
        hallucinatorResults += TREE_DATA.REWARDS[state]
    if stateOnlyTable in TREE_DATA.TERMINAL_STATES:
        tableResults += TREE_DATA.REWARDS[stateOnlyTable]

    if t % numTrials2Save == 0:
        endTime = datetime.datetime.now()
        print("num trials =", t, "time diff =", (endTime - startTime).total_seconds())
        TREE_DATA.Q_TABLE.to_pickle(qTableName + '.gz', 'gzip') 
        startTime = datetime.datetime.now()
    


hallucinatorResults = hallucinatorResults / countValueHallucinator[0]
for i in range(0, len(bestValueHallucinator)):
    if countValueHallucinator[i] != 0:
        bestValueHallucinator[i] = bestValueHallucinator[i] / countValueHallucinator[i]

tableResults = tableResults / countValueTable[0]
for i in range(0, len(bestValueTable)):
    if countValueTable[i] != 0:
        bestValueTable[i] = bestValueTable[i] / countValueTable[i]

TREE_DATA.Q_TABLE.to_pickle(qTableName + '.gz', 'gzip') 

plt.plot(bestValueTable)
plt.plot(bestValueHallucinator)

plt.legend(["table", "hallucinator"])
plt.savefig("hallucination_compare.png")

plt.figure()
resultDict = {}
resultDict["table"] = tableResults
resultDict["hallucinator"] = hallucinatorResults
plt.bar(resultDict.keys(), resultDict.values(), align='center')
plt.savefig("hallucination_results.png")
plt.show()

