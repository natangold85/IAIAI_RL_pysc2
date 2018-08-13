# build base sub agent
import random
import math
import os.path
import time
import sys

import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions


#decision makers
from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay

from utils_results import ResultFile
from utils_results import PlotMngr

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

# utils functions
from utils import EmptySharedData
from utils import SwapPnt
from utils import IsolateArea
from utils import GetScreenCorners
from utils import SelectBuildingValidPoint
from utils import IsValidPoint4Select
from utils import SelectUnitValidPoints
from utils import GetLocationForBuilding
from utils import PrintScreen

# possible types of play

QTABLE = 'q'
DQN = 'dqn'
USER_PLAY = 'play'

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "hist"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

class ACTIONS:

    ID_DO_NOTHING = 0
    ID_GATHER_MINERALS = 1
    ID_GATHER_GAS = 2
    ID_CREATE_SCV = 3

    NUM_ACTIONS = 4
    
    ACTION2STR = ["DoNothing" , "GatherMinerals", "GatherGas", "CreateSCV"]


class STATE:
    # state details
    MINERALS_BUCKETING = 25
    GAS_BUCKETING = 25

    COMMAND_CENTER_IDX = 0
    MINERALS_IDX = 1
    GAS_IDX = 2
    MINERALS_FIELDS_COUNT = 3
    GAS_REFINERY_COUNT = 4
    SCV_MINERALS = 5
    SCV_GAS = 6
    SCV_IDLE = 7
    SCV_BUILDING_QUEUE = 8
    
    SIZE = 9

    IDX2STR = ["ccNum" , "min", "gas", "min_fieldsCount", "gas_refCount", "scv_min", "scv_gas", "scv_idle", "scv_Q"]


ALL_TYPES = set([USER_PLAY, QTABLE, DQN])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[QTABLE][DIRECTORY] = "gatherResources_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "gather_dqn"
RUN_TYPES[DQN][DIRECTORY] = "gatherResources_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"

REWARD_NORMALIZATION = 100
GAS_REWARD_MULTIPLE = 2

class SharedDataGather(EmptySharedData):
    def __init__(self):
        super(SharedDataGather, self).__init__()
        self.scvIdle = 0
        self.scvMinerals = 0
        self.scvGas = 0
        self.activeRefinery = 0
        self.ScvBuildingQ = []

class ScvCmd:
    def __init__(self):
        self.m_stepsCounter = 0

class Gather(BaseAgent):
    def __init__(self, runArg = None, decisionMaker = None, isMultiThreaded = False):
        super(Gather, self).__init__()

        if runArg == None:
            runTypeArg = list(ALL_TYPES.intersection(sys.argv))
            runArg = runTypeArg.pop()
        self.agent = GatherResourcesSubAgent(runArg, decisionMaker, isMultiThreaded)

        self.maxNumRef = 2
        self.refineryMineralsCost = 75
    
    def GetDecisionMaker(self):
        return self.agent.GetDecisionMaker()

    def step(self, obs):
        super(Gather, self).step(obs)

        if obs.first():
            self.FirstStep(obs)

        a = self.agent.step(obs,self.moveNum, self.sharedData)

        self.currMin = obs.observation['player'][SC2_Params.MINERALS]

        if not self.startMineMinerals:
            sc2Action, terminal = self.StartMine(obs, self.moveNum)
        elif self.refBuildCmd + self.refBuildComplete != self.maxNumRef and self.currMin >= self.refineryMineralsCost:
            sc2Action, terminal = self.BuildRefinery(obs, self.moveNum)
        else:
            monitor = False
            
            if self.refBuildComplete != self.maxNumRef and self.refBuildCmd > 0:
                idx = np.random.randint(0,10)
                if idx > 6:
                    sc2Action, terminal = self.MonitorRefBuild(obs, self.moveNum)
                    monitor = True
            
            if not monitor:
                sc2Action, terminal = self.agent.Action2SC2Action(obs, a, self.moveNum)
                

        self.moveNum = 0 if terminal else self.moveNum + 1
        return sc2Action

    def FirstStep(self, obs):
        self.moveNum = 0
        self.startMineMinerals = False

        self.refBuildCmd = 0
        self.refBuildComplete = 0

        self.sharedData = SharedDataGather()
        self.sharedData.scvIdle = obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT]

    def StartMine(self, obs, moveNum):
        if moveNum == 0:
            return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_ALL]), False
        elif moveNum == 1:
            numScv = len(obs.observation['multi_select'])
            if numScv == 0:
                numScv = len(obs.observation['single_select'])
            
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.agent.GatherResource(SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] >= 0:

                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), False
        
        elif moveNum == 2:
            if obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT] == 0:
                self.sharedData.scvIdle = 0
                self.sharedData.scvMinerals = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]
                self.startMineMinerals = True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def BuildRefinery(self, obs, moveNum):
        if moveNum == 0:
            if obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT] > 0:
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_SINGLE]), False

            target = self.agent.FindClosestSCVFromRes(SC2_Params.NEUTRAL_MINERAL_FIELD)
                
            if target[0] >= 0:      
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), False
        
        elif moveNum == 1:
            finishedAction = False
            sc2Action = TerranUnit.BUILDING_SPEC[Terran.Refinery].sc2Action

            if sc2Action in obs.observation['available_actions']:
                self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
                coord = GetLocationForBuilding(obs, self.cameraCornerNorthWest, self.cameraCornerSouthEast, Terran.Refinery)

                if coord[SC2_Params.Y_IDX] >= 0:
                    self.refBuildCmd += 1
                    self.sharedData.scvMinerals -= 1
                    return actions.FunctionCall(sc2Action, [SC2_Params.NOT_QUEUED, SwapPnt(coord)]), finishedAction

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def MonitorRefBuild(self, obs, moveNum):
        if moveNum == 0:
            target = SelectBuildingValidPoint(obs.observation['feature_screen'][SC2_Params.UNIT_TYPE], Terran.Refinery)
            if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), False

        elif moveNum == 1:
            refNum = self.RefineryNum(obs)
            if self.refBuildComplete < refNum:
                cmdEnded = refNum - self.refBuildComplete
                self.refBuildCmd -= cmdEnded
                self.refBuildComplete = refNum
                self.sharedData.activeRefinery = refNum
                self.sharedData.scvGas += cmdEnded

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def RefineryNum(self, obs):
        buildingStatus = obs.observation['multi_select']
        if len(buildingStatus) == 0:
            buildingStatus = obs.observation['single_select']
            if len(buildingStatus) == 0:
                return self.refBuildComplete

        if buildingStatus[0][SC2_Params.UNIT_TYPE_IDX] != Terran.Refinery:
            return self.refBuildComplete

        numComplete = 0
        for stat in buildingStatus[:]:
            if stat[SC2_Params.BUILDING_COMPLETION_IDX] == 0:
                numComplete += 1
        
        return numComplete

class GatherResourcesSubAgent:
    def __init__(self, runArg, decisionMaker = None, isMultiThreaded = False):        
        runType = RUN_TYPES[runArg]

        self.illigalmoveSolveInModel = True
        # tables:
        if decisionMaker == None:
            if runType[TYPE] != "play":
                self.decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                                        resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=runType[DIRECTORY], isMultiThreaded=isMultiThreaded)
            else:
                self.illigalmoveSolveInModel = False
                self.decisionMaker = UserPlay(False, numActions = ACTIONS.NUM_ACTIONS)
        else:
            self.decisionMaker = decisionMaker

        # states and action:
        self.terminalState = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        # model params
        self.unit_type = None

        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]

        self.scvPriceMinerals = 50
    
    def GetDecisionMaker(self):
        return self.decisionMaker

    def step(self, obs, moveNum = 0, sharedData = None):  

        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        
        if obs.first():
            self.FirstStep(obs)
        elif obs.last():
            self.LastStep(obs)

        if sharedData != None:
            self.sharedData = sharedData

        if moveNum == 0: 
            self.CreateState(obs)
            self.Learn()
            self.current_action = self.ChooseAction()
            #self.PrintState()

        self.numSteps += 1

        return self.current_action

    def Action2SC2Action(self, obs, a, moveNum):
        self.isActionCommitted = True

        if a == ACTIONS.ID_DO_NOTHING:
            return SC2_Actions.DO_NOTHING_SC2_ACTION, True
        elif a == ACTIONS.ID_GATHER_GAS:
            return self.GatherGas(obs, moveNum)
        elif a == ACTIONS.ID_GATHER_MINERALS:
            return self.GatherMinerals(obs, moveNum)
        elif a == ACTIONS.ID_CREATE_SCV:
            return self.CreateScv(obs, moveNum)
            

    def FirstStep(self, obs):
        self.numSteps = 0
        self.moveNum = 0

        self.current_action = None
        self.previous_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        self.sharedData = SharedDataGather()
        self.takeScvFrom = STATE.SCV_IDLE

        self.isActionCommitted = False

        self.currWorkerCount = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]
        self.maxSupply = False

        self.accumulatedReward = 0.0

    def LastStep(self, obs, reward = 0):
        reward = obs.reward
        score = obs.observation["score_cumulative"][0]
        self.decisionMaker.end_run(self.accumulatedReward, score, self.numSteps)



    def Learn(self, reward = 0):
        if self.isActionCommitted:
            r = reward + self.CalcReward()
            self.accumulatedReward += r
            # print("\n\nreward =", r, "curr minerals =", self.current_state[STATE.MINERALS_IDX], "curr gas =", self.current_state[STATE.GAS_IDX], "prev minerals =", 
            #                                 self.previous_state[STATE.MINERALS_IDX], "prev gas =", self.previous_state[STATE.GAS_IDX] )
            self.decisionMaker.learn(self.previous_scaled_state, self.current_action, reward, self.current_scaled_state)
        
        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def IsDoNothingAction(self, a):
        return a == ACTIONS.ID_DO_NOTHING

    def GatherGas(self, obs, moveNum):
        if moveNum == 0:
            finishedAction = False
            if self.current_scaled_state[STATE.SCV_IDLE] > 0:
                self.takeScvFrom = STATE.SCV_IDLE
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_SINGLE]), finishedAction
            elif STATE.SCV_MINERALS > 0:
                self.takeScvFrom = STATE.SCV_MINERALS
                target = self.FindClosestSCVFromRes(SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction

        elif moveNum == 1:
            finishedAction = True
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.GatherResource([Terran.Refinery] + SC2_Params.VESPENE_GAS_FIELD )
                if target[0] >= 0:
                    self.sharedData.scvGas += 1
                    if self.takeScvFrom == STATE.SCV_MINERALS:
                        self.sharedData.scvMinerals = max(0, self.sharedData.scvMinerals - 1)

                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction
        
     
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def GatherMinerals(self, obs, moveNum):
        if moveNum == 0:
            finishedAction = False
            if self.current_scaled_state[STATE.SCV_IDLE] > 0 and SC2_Actions.SELECT_IDLE_WORKER in obs.observation['available_actions']:
                self.takeScvFrom = STATE.SCV_IDLE
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_SINGLE]), finishedAction
            elif STATE.SCV_GAS > 0:
                target = self.FindClosestSCVFromRes([Terran.Refinery] + SC2_Params.VESPENE_GAS_FIELD)
                if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                    self.takeScvFrom = STATE.SCV_GAS
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction

        elif moveNum == 1:
            finishedAction = True
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.GatherResource(SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] >= 0:
                    self.sharedData.scvMinerals += 1
                    if self.takeScvFrom == STATE.SCV_GAS:
                        self.sharedData.scvGas = max(0, self.sharedData.scvGas - 1)

                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True  

    def CreateScv(self, obs, moveNum):
        if moveNum == 0:
            target = SelectBuildingValidPoint(self.unit_type, Terran.CommandCenter)
            if target[0] >= 0:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), False
        elif moveNum == 1:
            if SC2_Actions.TRAIN_SCV in obs.observation['available_actions']:
                self.sharedData.ScvBuildingQ.append(ScvCmd())
                return actions.FunctionCall(SC2_Actions.TRAIN_SCV, [SC2_Params.QUEUED]), True
        
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True 

    def CreateState(self, obs):    
        self.current_state[STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        self.current_state[STATE.COMMAND_CENTER_IDX] = 1
        
        self.current_state[STATE.SCV_IDLE] = obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT]
        self.current_state[STATE.SCV_MINERALS] = self.sharedData.scvMinerals
        self.current_state[STATE.SCV_GAS] = self.sharedData.scvGas

        workerCount = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]
        if workerCount > self.currWorkerCount:
            newWorkers = workerCount - self.currWorkerCount
            if len(self.sharedData.ScvBuildingQ) > newWorkers:
                for i in range(newWorkers):
                    self.sharedData.ScvBuildingQ.pop(0)
            else:
                self.sharedData.ScvBuildingQ = []

        self.current_state[STATE.SCV_BUILDING_QUEUE] = len(self.sharedData.ScvBuildingQ)

        min_y, min_x = self.AllResouresPnts(SC2_Params.NEUTRAL_MINERAL_FIELD) 
        self.current_state[STATE.MINERALS_FIELDS_COUNT] = len(min_y)
        self.current_state[STATE.GAS_REFINERY_COUNT] = self.sharedData.activeRefinery

        self.ScaleState()

        self.maxSupply = obs.observation['player'][SC2_Params.SUPPLY_USED] == obs.observation['player'][SC2_Params.SUPPLY_CAP]

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[STATE.MINERALS_IDX] = int(self.current_scaled_state[STATE.MINERALS_IDX] / STATE.MINERALS_BUCKETING) * STATE.MINERALS_BUCKETING
        self.current_scaled_state[STATE.GAS_IDX] = int(self.current_scaled_state[STATE.GAS_IDX] / STATE.GAS_BUCKETING) * STATE.GAS_BUCKETING

    def ChooseAction(self):
        if self.illigalmoveSolveInModel:
            validActions = self.ValidActions()
            if np.random.uniform() > self.decisionMaker.ExploreProb() :
                valVec = self.decisionMaker.ActionValuesVec(self.current_state)   
                random.shuffle(validActions)
                validVal = valVec[validActions]
                action = validActions[validVal.argmax()]
            else:
                action = np.random.choice(validActions) 
        else:
            #print("valid actions = ", self.ValidActions())
            action = self.decisionMaker.choose_action(self.current_state)
        
        return action

    def ValidActions(self):
        valid = [ACTIONS.ID_DO_NOTHING]

        numIdle = self.current_scaled_state[STATE.SCV_IDLE]
        numGas = self.current_scaled_state[STATE.SCV_GAS]
        numMinerals = self.current_scaled_state[STATE.SCV_MINERALS]
        
        if numIdle + numGas > 0 and self.current_scaled_state[STATE.MINERALS_FIELDS_COUNT] > 0:
            valid.append(ACTIONS.ID_GATHER_MINERALS)

        if numIdle + numMinerals > 0 and self.current_scaled_state[STATE.GAS_REFINERY_COUNT] > 0:
            valid.append(ACTIONS.ID_GATHER_GAS)

        if self.current_scaled_state[STATE.MINERALS_IDX] >= self.scvPriceMinerals and not self.maxSupply:
            valid.append(ACTIONS.ID_CREATE_SCV)
        
        return valid


    def GatherResource(self, resourceList):
        res_y = []
        res_x = []
        resType = []
        resMaps = {}
        for val in resourceList:
            p_y, p_x = (self.unit_type == val).nonzero()
            res_y += list(p_y)
            res_x += list(p_x)
            resType += [val] * len(p_y)
            resMaps[val] = (self.unit_type == val)

        if len(res_y) > 0:
            validPnt = False
            while not validPnt:
                i = np.random.randint(0, len(res_y))

                validPnt =  IsValidPoint4Select(resMaps[resType[i]], res_y[i], res_x[i])

            return [res_y[i], res_x[i]]
        
        return [-1,-1]

    def AllResouresPnts(self, resourceList):
        res_y = []
        res_x = []
        for val in resourceList:
            p_y, p_x = (self.unit_type == val).nonzero()
            res_y += list(p_y)
            res_x += list(p_x)
        return res_y, res_x
    
    def RefineryCount(self):
        onlyBuilding = self.unit_type == Terran.Refinery
        buildingAndResource = onlyBuilding
        for res in SC2_Params.VESPENE_GAS_FIELD:
            buildingAndResource = buildingAndResource | (self.unit_type == res)

        b_y, b_x = onlyBuilding.nonzero()
        b_y = list(b_y)
        b_x = list(b_x)
        
        refNum = 0
        
        while len(b_y) > 0:
            loc = [b_y[0], b_x[0]]
            building_y, building_x = IsolateArea(loc, buildingAndResource)
            #print("num pts =", len(r_y), "num 1 building =", len(building_y),end = ', ')
            toRemove = []
            for b in range(len(building_y)):
                for p in range(len(b_y)):
                    if b_y[p] == building_y[b] and b_x[p] == building_x[b]:
                        if p not in toRemove:
                            toRemove.append(p)
                        break

            toRemove = sorted(toRemove)
            for i in range(len(toRemove)):
                del b_y[toRemove[i] - i]
                del b_x[toRemove[i] - i]
                                
            
            refNum += 1
        
        return refNum

    def Action2Str(self, a):
        return ACTIONS.ACTION2STR[a]


    def FindClosestSCVFromRes(self, resList):
        minDist = 10000
        idxMin = -1
        scv_y, scv_x = SelectUnitValidPoints(self.unit_type == Terran.SCV)
        for resVal in resList:
            res_y, res_x = (self.unit_type == resVal).nonzero()
            for s in range(len(scv_y)):
                for r in range(len(res_y)):
                    diffY = res_y[r] - scv_y[s]
                    diffX = res_x[r] - scv_x[s]
                    dist = diffY * diffY + diffX * diffX
                    if dist < minDist:
                        minDist = dist
                        idxMin = s

        if idxMin >= 0:
            return [scv_y[idxMin], scv_x[idxMin]]
        else:
            return [-1,-1]


    def PrintState(self):
        if self.current_action != None:
            print("action =", self.Action2Str(self.current_action), end = "\n [ ")
        for i in range(STATE.SIZE):
            print(STATE.IDX2STR[i], self.current_state[i], end = ', ')
        print(']')
    
    def CalcReward(self):
        diffMin = self.current_state[STATE.MINERALS_IDX] - self.previous_state[STATE.MINERALS_IDX]
        diffGas = self.current_state[STATE.GAS_IDX] - self.previous_state[STATE.GAS_IDX]

        return (diffMin + diffGas * GAS_REWARD_MULTIPLE) / REWARD_NORMALIZATION

    def CheckScvSelected(self, obs):
        select = obs.observation['single_select']
        if len(select) == 0:
            return False

        return select[0][SC2_Params.UNIT_TYPE_IDX] == Terran.SCV

if __name__ == "__main__":
    if "results" in sys.argv:
        runTypeArg = list(ALL_TYPES.intersection(sys.argv))
        runTypeArg.sort()
        resultFnames = []
        directoryNames = []
        for arg in runTypeArg:
            runType = RUN_TYPES[arg]
            fName = runType[RESULTS]
            
            if DIRECTORY in runType.keys():
                dirName = runType[DIRECTORY]
            else:
                dirName = ''

            resultFnames.append(fName)
            directoryNames.append(dirName)

        grouping = int(sys.argv[len(sys.argv) - 1])
        plot = PlotMngr(resultFnames, directoryNames, runTypeArg)
        plot.Plot(grouping)