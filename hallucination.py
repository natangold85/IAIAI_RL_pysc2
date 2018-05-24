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
        self.DISCOUNT = 0.95
        
        self.EXPLORE_STOP = 1
        self.EXPLORE_START = 0.1
        self.EXPLORE_RATE = 0.1


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
            self.UpdateQTable()
            self.TrimRoot(state, prevAction)
        else:
            self.currRoot = StateNode(state)


        startTime = datetime.datetime.now()
        currTime = datetime.datetime.now()

        while (currTime - startTime).total_seconds() < self.searchDuration:
            self.currRoot.Simulate()
			
            currTime = datetime.datetime.now()

    def UpdateQTable(self):
        a = 9

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
    
    def ExecuteAction(self, state, action):
        return self.currRoot.FindNextState(state, action)

  
class StateNode:
    def __init__(self, state, count = 0):   
        self.state = state 
        self.count = count 
        self.actionChilds = []
    
    def Simulate(self, currIdx = 0):
        if len(self.actionChilds) == 0:
            self.CreateActionChilds()

        action = self.UpperBoundAction()
        print("action chosen =", action)

        nextState = self.FindNextState(self.state, action)

        print("real depth =", currIdx, "simulate depth =", self.Depth(), "state =", self.state, "\n\tnext state =", nextState)
        time.sleep(0.5)
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
        # print("\trollout depth =", currIdx)
        action = TREE_DATA.Q_TABLE.ix[state, :].idxmax()
        nextState = self.FindNextState(state, action)
        print(state, " -> ", nextState)
        if nextState in TREE_DATA.TERMINAL_STATES:
            print("rollout terminal")
            return TREE_DATA.REWARDS[nextState]
        else:
            return TREE_DATA.DISCOUNT * self.Rollout(nextState, currIdx + 1)


    def CreateActionChilds(self):
        valueVector = TREE_DATA.Q_TABLE.ix[self.state, :]
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
class ActionNode:
    def __init__(self, state, action, value = 0):
        self.fatherState = state
        self.action = action
        self.stateChilds = {}
        self.value = value
        self.count = 0
    
    def AddState(self, state, count = 0):
        self.stateChilds[state] = StateNode(state, count)
            
    def UpdateReward(self, reward):
        self.value = (self.value * self.count + reward) / (self.count + 1)
        self.count += 1

    def Depth(self):
        d = 0
        for key, state in self.stateChilds.items():
            d = max(d, state.Depth())

        return d + 1

if len(sys.argv) < 3:
    print("enter table names")

qTableName = sys.argv[1]
tTableName = sys.argv[2]
numActions = int(sys.argv[3])

qtable = pd.read_pickle(qTableName + '.gz', compression='gzip')

#print(qtable[qtable.index != "TrialsData"].describe())
ttable = pd.read_pickle(tTableName + '.gz', compression='gzip')
treeMngr = TreeMngr(ttable, qtable, numActions, 5)

numVals = 0
while numVals < 50:
    state = random.choice(list(ttable.keys()))
    numVals = sum(ttable[state][1])

state = "[4 0 0 0 0 4 0 4 1 9]"
print("state chosen =", state, "action val =", ttable[state][1])
print(ttable[state][0], "\n\n\n")
print("qtable vals =\n" , qtable.ix[state,:])
exit()


action = None
for i in range(0, 1):
    treeMngr.Hallucinate(state, action)
    print("for state =", state, "action chosen =", action)
    # action = treeMngr.ChooseAction()
    # qTableState = TTableState2QTableState(state)
    # print("state for qtable =", qTableState)
    print("qtable vals =\n" , qtable.ix[state,:])
    # treeMngr.ExecuteAction(state, action) 