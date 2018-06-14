
import time
import random
import numpy as np
import pandas as pd
import math
import sys
import os.path
import matplotlib
import matplotlib.pyplot as plt

TTABLE_IDX_ACTIONS_COUNT = 1
TTABLE_IDX_TABLE = 0

class TreeData:
    def __init__(self):
        # solver params
        self.LEARNING_RATE = 0.01
        
        self.DISCOUNT = 0.95
        
        self.EXPLORE_STOP = 0.01
        self.EXPLORE_START = 1.0
        self.EXPLORE_RATE = 0.0001

        self.SIZE_TRANSITION_TO_COMPLETE_TABLE = 5
        # tables
        self.MIN_TRANSITIONS_TO_CONTINUE_SIM = 1
        self.T_TABLE = None
        self.Q_TABLE = None
        self.STATE_DICT = None
        self.VALUES_2_UPDATE_DICT = None

        self.MAX_SIMULATE_DEPTH = 25
        self.MAX_ROLLOUT_DEPTH = 300

        # model params
        self.NUM_ACTIONS = None
        self.TERMINAL_STATES = ['win', 'loss', 'tie', 'terminal']
        
        self.REWARDS = {}
        self.REWARDS['win'] = 1
        self.REWARDS['loss'] = -1
        self.REWARDS['tie'] = -1
        self.REWARDS['terminal'] = 0

        self.MIN_REWARD = min(self.REWARDS.values())

    def ExplorationFactor(self, count):
        return self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP) * np.exp(-self.EXPLORE_RATE * count)

    def ChooseRandomState(self, minTransitionsCount = 5):
        numVals = 0
        while numVals < minTransitionsCount:
            state = random.choice(list(self.T_TABLE.table.keys()))
            if type(self.T_TABLE.table[state])==list and len(self.T_TABLE.table[state]) > 1:
                numVals = sum(self.T_TABLE.table[state][1])
        stateOnlyTable = ''.join(state)
        return stateOnlyTable

    def UpdateQTable(self, qTableName):
        self.Q_TABLE = pd.read_pickle(qTableName + '.gz', compression='gzip')
        count = 0
        for s, toUpdate in self.VALUES_2_UPDATE_DICT.items():
            count += len(toUpdate)
            for a, val in toUpdate.items():
                self.Q_TABLE.ix[s,a] += self.LEARNING_RATE * (val - self.Q_TABLE.ix[s,a] )
        
        self.Q_TABLE.to_pickle(qTableName + '.gz', 'gzip')

        return count

TREE_DATA = TreeData()

def LenStat(tTableName, numiterations = 100, maxNumRec = 100, runOnAllOptions = False):
    ttable = pd.read_pickle(tTableName + '.gz', compression='gzip')
    terminalStates = ['loss' ,'win','tie']
    allStates = list(ttable.keys())
    allLength = []
    allDeadEnds = 0
    allLiveEnds = 0
    i = 0
    while i < numiterations:
        s = random.choice(allStates)
        if s != "TrialsData":
            print("\n\n\nstate:",s,"\n", ttable[s])
            i += 1
            l, deadEnds, liveEnds = LenToTerminal(ttable, s, terminalStates, runOnAllOptions, maxNumRec)
            allLength = l + allLength
            allDeadEnds = deadEnds + allDeadEnds
            allLiveEnds = allLiveEnds + liveEnds

    print("deadEnds = ", allDeadEnds, "liveEnds = ", allLiveEnds)
    plt.hist(allLength)
    plt.show()
    return allLength, allDeadEnds, allLiveEnds

def LenToTerminal(ttable, s, terminalStates, runOnAllOptions = False, maxRecDepth = 50, currIdx = 0):
    if s in terminalStates:
        return [currIdx], 0, 1
    elif currIdx == maxRecDepth:
        return [currIdx], 1, 0
    else:
        #stateHist.append(s)
        table = ttable[s][0]
        allStates = list(table.index)
        allLength = []

        allLiveEnds = 0       
        if len(allStates) == 0:
            allDeadEnds = 1
        else:
            allDeadEnds = 0
            
            if runOnAllOptions:
                for s_ in allStates:
                    if s_ != s:
                        l, deadEnds, liveEnds = LenToTerminal(ttable, s_, terminalStates, runOnAllOptions, maxRecDepth, currIdx + 1)
                        allLength = allLength + l
                        allDeadEnds = deadEnds + allDeadEnds
                        allLiveEnds = liveEnds + allLiveEnds
            else:
                choose = False
                while not choose:
                    s_ = random.choice(allStates)
                    if s_ != s:
                        choose = True

                allLength, allDeadEnds, allLiveEnds = LenToTerminal(ttable, s_, terminalStates, runOnAllOptions, maxRecDepth, currIdx + 1)

        return allLength, allDeadEnds, allLiveEnds



# LenStat("melee_attack_ttable_onlineHallucination", 20,50, True)


def HallucinationMngrPSFunc(sharedDict):
    TREE_DATA.NUM_ACTIONS = sharedDict["num_actions"]
    TREE_DATA.STATE_DICT = {}
    TREE_DATA.VALUES_2_UPDATE_DICT = {}
    qTableName = sharedDict["q_table"]
    tTableName = sharedDict["t_table"]
    
    hMngr = HallucinationMngr()

    TTableCreated = False

    if os.path.isfile(tTableName + '.gz') and not TTableCreated:
        TREE_DATA.T_TABLE = pd.read_pickle(tTableName + '.gz', compression='gzip')
        TTableCreated = True
    
    while True:
        while TTableCreated:
            if hMngr.UpdateRoot(sharedDict):
                hMngr.Hallucinate(sharedDict)
                hMngr.InsertValues2Dict()

            if sharedDict["updateTableFlag"]:
                break

        updateTableFlag = False
        while not updateTableFlag:
            updateTableFlag = sharedDict["updateTableFlag"]
        
        count = TREE_DATA.UpdateQTable(qTableName)
        TREE_DATA.T_TABLE = pd.read_pickle(tTableName + '.gz', compression='gzip')
        TTableCreated = True
        sharedDict["updateTableFlag"] = False
        print("\n\n\tupdate q table count vals =", count)
        TREE_DATA.VALUES_2_UPDATE_DICT = {}
        TREE_DATA.STATE_DICT = {}

    



class HallucinationMngr:
    def __init__(self):
        self.currRoot = None

    def UpdateTables(self, qTable, tTable):
        TREE_DATA.Q_TABLE = qTable.table.copy()
        TREE_DATA.T_TABLE = tTable.table.copy()
        TREE_DATA.STATE_DICT = {}
        self.updateTable = True

    def Hallucinate(self, sharedDict):
        contHallucinate = True

        while contHallucinate:
            self.currRoot.Simulate(self.currRoot)
            if  sharedDict["updateStateFlag"] == True:
                contHallucinate = False

    def UpdateRoot(self, sharedDict):
        
        if sharedDict["updateStateFlag"] == True:
            nextState = sharedDict["nextState"]
            sharedDict["updateStateFlag"] = False
            if nextState in TREE_DATA.T_TABLE:
                if nextState not in TREE_DATA.STATE_DICT:
                    TREE_DATA.STATE_DICT[nextState] = StateNode(nextState)
                
                self.currRoot = TREE_DATA.STATE_DICT[nextState]
                return True   
                  
        return False

    def ChooseAction(self):
        if self.currRoot == None:
            return None
        
        return self.currRoot.ChooseAction()

    def Value(self,action):
        if self.currRoot == None:
            return None
        
        return self.currRoot.Value(action)
        
    def ExecuteAction(self, state, action):
        if state in TREE_DATA.T_TABLE:
            return self.currRoot.FindNextState(state, action)
        else:
            return "terminal"

    def InsertValues2Dict(self):
        if self.currRoot != None:
            self.currRoot.InsertValues2Dict()

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

    def Count(self):
        if self.currRoot != None:
            return self.currRoot.count
        else:
            return 0

  
class StateNode:
    def __init__(self, state, count = 0):   
        self.state = state 
        self.count = count 
        self.actionChilds = []
        self.CreateActionChilds()

        # self.terminationCount = {}

        # self.terminationCount["loss"] = 0
        # self.terminationCount["win"] = 0
        # self.terminationCount["tie"] = 0
        # self.terminationCount["terminal"] = 0
        # self.terminationCount["recDepth"] = 0
    
    def Simulate(self, root, currIdx = 0):
        action = self.UpperBoundAction()
        nextState = self.FindNextState(self.state, action)

        if nextState in TREE_DATA.TERMINAL_STATES:
            maxIdx = currIdx
            reward = TREE_DATA.REWARDS[nextState]
        elif nextState not in self.actionChilds[action].stateChilds or currIdx >= TREE_DATA.MAX_SIMULATE_DEPTH:
            maxIdx = currIdx
            rollOutHist = [nextState]
            reward = TREE_DATA.DISCOUNT * self.Rollout(nextState, root, rollOutHist, currIdx + 1)
            self.actionChilds[action].AddState(nextState)
        else:
            r, maxIdx = self.actionChilds[action].stateChilds[nextState].Simulate(root, currIdx + 1)
            reward = TREE_DATA.DISCOUNT * r

        self.UpdateReward(action, reward)
        
        return reward, maxIdx
        
    def Rollout(self, state, root, rolloutHist, currIdx):
        if currIdx >= TREE_DATA.MAX_ROLLOUT_DEPTH:
            return 0
        
        rolloutHist.append(state)

        nextState = "terminal"
        if state in TREE_DATA.T_TABLE:
            if TREE_DATA.T_TABLE[state][TTABLE_IDX_ACTIONS_COUNT] != None:
                actionCount = TREE_DATA.T_TABLE[state][TTABLE_IDX_ACTIONS_COUNT]
                existingAction = [i for i in range(len(actionCount)) if actionCount[i] > 0]
                action = random.choice(existingAction)
                nextState = self.FindNextState(state, action, rolloutHist)
        

        if nextState in TREE_DATA.TERMINAL_STATES:
            return TREE_DATA.REWARDS[nextState]
        else:
            return TREE_DATA.DISCOUNT * self.Rollout(nextState, root, rolloutHist, currIdx + 1)


    def CreateActionChilds(self):   
        valueVector = np.zeros(TREE_DATA.NUM_ACTIONS, dtype=np.float, order='C')

        for a in range(0, TREE_DATA.NUM_ACTIONS):
            self.actionChilds.append(ActionNode(self.state, a, valueVector[a]))
            
        

    def UpperBoundAction(self):
        bestValue = TREE_DATA.MIN_REWARD - 1
        bestAction = -1
        actionCount = TREE_DATA.T_TABLE[self.state][TTABLE_IDX_ACTIONS_COUNT]
        existingActions = [i for i in range(len(actionCount)) if actionCount[i] > 0]
        for a in existingActions:
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
        
    def InsertValues2Dict(self):
        if self.state not in TREE_DATA.VALUES_2_UPDATE_DICT:
            TREE_DATA.VALUES_2_UPDATE_DICT[self.state] = {}

        for a in range(0, TREE_DATA.NUM_ACTIONS):
            child = self.actionChilds[a]
            if child.count > 0:
                TREE_DATA.VALUES_2_UPDATE_DICT[self.state][a] = child.value 


    def FindNextState(self, state, action, avoidStates = []):
        stateSize = TREE_DATA.T_TABLE[state][TTABLE_IDX_ACTIONS_COUNT][action]
        stateNum = random.randint(0, stateSize)
        
        prevDifferentState = "terminal"
        for s_ in TREE_DATA.T_TABLE[state][TTABLE_IDX_TABLE].index:
            stateNum -= TREE_DATA.T_TABLE[state][TTABLE_IDX_TABLE].ix[s_, action]
            if s_ != state and s_ not in avoidStates:
                prevDifferentState = s_
            if stateNum <= 0 and prevDifferentState != "terminal":
                break
        return prevDifferentState




    def UpdateReward(self, action, reward):
        if reward != 0:
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
        
        return size + 1      

class ActionNode:
    def __init__(self, state, action, value = 0):
        self.fatherState = state
        self.action = action
        self.stateChilds = {}
        self.value = value
        self.count = 0
    
    def AddState(self, state, count = 0):
        if state not in TREE_DATA.STATE_DICT:
            TREE_DATA.STATE_DICT[state] = StateNode(state, count)
        
        if state != self.fatherState:
            self.stateChilds[state] = TREE_DATA.STATE_DICT[state]
            
    def UpdateReward(self, reward):
        if reward != 0:
            self.value = (self.value * self.count + reward) / (self.count + 1)
            self.count += 1

    def Depth(self):
        d = 0
        for state in self.stateChilds.values():
            d = max(d, state.Depth())

        return d

    def Size(self):
        size = len(self.stateChilds)
        for state in self.stateChilds.values():
            size += state.Size()
        
        return size   

