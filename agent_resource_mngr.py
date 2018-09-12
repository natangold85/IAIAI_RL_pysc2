# build base sub agent
import random
import math
import os.path
import time
import sys

import numpy as np
import pandas as pd

from multiprocessing import Lock


from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions


#decision makers
from utils_decisionMaker import BaseNaiveDecisionMaker
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
from utils import GatherResource
from utils import EmptySharedData
from utils import SwapPnt
from utils import IsolateArea
from utils import GetScreenCorners
from utils import SelectBuildingValidPoint
from utils import IsValidPoint4Select
from utils import SelectUnitValidPoints
from utils import GetLocationForBuilding
from utils import PrintScreen

AGENT_DIR = "ResourceMngr/"
AGENT_NAME = "resource_mngr"

REWARD_NORMALIZATION = 100
GAS_REWARD_MULTIPLE = 2

SCV_GROUP_MINERALS = 0
SCV_GROUP_GAS1 = 1
SCV_GROUP_GAS2 = 2
GAS_GROUPS = [SCV_GROUP_GAS1, SCV_GROUP_GAS2]
ALL_SCV_GROUPS = [SCV_GROUP_MINERALS] + GAS_GROUPS

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
    ID_ADD2_MINERALS_FROM_GAS1 = 1
    ID_ADD2_MINERALS_FROM_GAS2 = 2
    ID_ADD2_GAS1_FROM_MINERALS = 3
    ID_ADD2_GAS1_FROM_GAS2 = 4
    ID_ADD2_GAS2_FROM_MINERALS = 5
    ID_ADD2_GAS2_FROM_GAS1 = 6
    ID_CREATE_SCV = 7

    NUM_ACTIONS = 8
    
    ACTION2STR = ["DoNothing" , "Add2MineralsFromGas1", "Add2MineralsFromGas2", "Add2Gas1FromMinerals", "Add2Gas1FromGas2", "Add2Gas2FromMinerals", "Add2Gas2FromGas1", "CreateSCV"]

    GROUP2ACTION = np.ones((len(ALL_SCV_GROUPS), len(ALL_SCV_GROUPS)), int) * NUM_ACTIONS

    GROUP2ACTION[SCV_GROUP_MINERALS, SCV_GROUP_GAS1] = ID_ADD2_MINERALS_FROM_GAS1
    GROUP2ACTION[SCV_GROUP_MINERALS, SCV_GROUP_GAS2] = ID_ADD2_MINERALS_FROM_GAS2
    
    GROUP2ACTION[SCV_GROUP_GAS1, SCV_GROUP_MINERALS] = ID_ADD2_GAS1_FROM_MINERALS
    GROUP2ACTION[SCV_GROUP_GAS1, SCV_GROUP_GAS2] = ID_ADD2_GAS1_FROM_GAS2

    GROUP2ACTION[SCV_GROUP_GAS2, SCV_GROUP_MINERALS] = ID_ADD2_GAS2_FROM_MINERALS
    GROUP2ACTION[SCV_GROUP_GAS2, SCV_GROUP_GAS1] = ID_ADD2_GAS2_FROM_GAS1

class RESOURCE_STATE:
    # state details
    MINERALS_MAX = 500
    GAS_MAX = 300

    MINERALS_BUCKETING = 50
    GAS_BUCKETING = 50

    SCV_MINERALS_PRICE = 50

    MINERALS_IDX = 0
    GAS_IDX = 1
    MINERALS_FIELDS_COUNT = 2
    GAS_REFINERY_COUNT = 3
    SCV_GROUP_MINERALS_IDX = 4
    SCV_GROUP_GAS1_IDX = 5
    SCV_GROUP_GAS2_IDX = 6
    SCV_BUILDING_QUEUE = 7
    
    SIZE = 8

    IDX2STR = ["ccNum" , "min", "gas", "min_fieldsCount", "gas_refCount", "scv_min", "scv_gas1", "scv_gas2", "scv_Q"]

    GROUP2IDX = {}
    GROUP2IDX[SCV_GROUP_MINERALS] = SCV_GROUP_MINERALS_IDX
    GROUP2IDX[SCV_GROUP_GAS1] = SCV_GROUP_GAS1_IDX
    GROUP2IDX[SCV_GROUP_GAS2] = SCV_GROUP_GAS2_IDX


ALL_TYPES = set([USER_PLAY, QTABLE, DQN])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(RESOURCE_STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[QTABLE][DIRECTORY] = "gatherResources_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(RESOURCE_STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "gather_dqn"
RUN_TYPES[DQN][DIRECTORY] = "gatherResources_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"

class SharedDataResourceMngr(EmptySharedData):
    def __init__(self):
        super(SharedDataResourceMngr, self).__init__()
        self.scvBuildingQ = []
        self.scvMineralGroup = None
        self.scvGasGroups = []
        for i in GAS_GROUPS:
            self.scvGasGroups.append(None)

class ScvCmd:
    def __init__(self):
        self.m_stepsCounter = 0



class NaiveDecisionMakerResource(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Learn = 20):
        super(NaiveDecisionMakerResource, self).__init__(numTrials2Learn, resultFName, directory)
        self.desiredGroupCnt = {}
        self.desiredGroupCnt[SCV_GROUP_MINERALS] = 8
        self.desiredGroupCnt[SCV_GROUP_GAS1] = 3
        self.desiredGroupCnt[SCV_GROUP_GAS2] = 3

        self.desiredScvSize = 8 + 3+ 3
        self.maxQSize = 5

    def choose_action(self, state):
        hasRes = state[RESOURCE_STATE.MINERALS_IDX] > RESOURCE_STATE.SCV_MINERALS_PRICE
        minSize = state[RESOURCE_STATE.SCV_GROUP_MINERALS_IDX]
        qSize = state[RESOURCE_STATE.SCV_BUILDING_QUEUE]

        gas1Size = state[RESOURCE_STATE.SCV_GROUP_GAS1_IDX]
        gas2Size = state[RESOURCE_STATE.SCV_GROUP_GAS2_IDX]
        
        sumScv = minSize + gas1Size + gas2Size + qSize

        if minSize > self.desiredGroupCnt[SCV_GROUP_MINERALS]:
            if gas1Size < self.desiredGroupCnt[SCV_GROUP_GAS1]:
                return ACTIONS.ID_ADD2_GAS1_FROM_MINERALS
            else:
                return ACTIONS.ID_ADD2_GAS2_FROM_MINERALS
        elif sumScv < self.desiredScvSize and hasRes:
            return ACTIONS.ID_CREATE_SCV
        
        return ACTIONS.ID_DO_NOTHING

    def GroupSmaller(self, state):
        for group, stateIdx in RESOURCE_STATE.GROUP2IDX.items():
            if state[stateIdx] < self.desiredGroupCnt[group]:
                return group, self.desiredGroupCnt[group] - state[stateIdx]
        
        return -1, 0
    
    def GroupBigger(self, state):
        for group, stateIdx in RESOURCE_STATE.GROUP2IDX.items():
            if state[stateIdx] > self.desiredGroupCnt[group]:
                return group
        return -1

    def ActionValuesVec(self, state, target = True):    
        vals = np.zeros(ACTIONS.NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state)] = 1.0

        return vals



class ResourceMngrSubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):
        super(ResourceMngrSubAgent, self).__init__(RESOURCE_STATE.SIZE)

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.current_action = None

        self.sharedData = sharedData

        self.scvMineralPrice = 50
        self.rallyCoordScv = [50,50]

        # required num scv
        self.numScvReq4Group = {}
        self.numScvReq4Group[SCV_GROUP_MINERALS] = 8
        self.numScvReq4Group[SCV_GROUP_GAS1] = 3
        self.numScvReq4Group[SCV_GROUP_GAS2] = 3

        self.numScvRequired = 0
        for req in self.numScvReq4Group.values():
            self.numScvRequired += req

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        decisionMaker = NaiveDecisionMakerResource()
        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker


    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1
        
    def FirstStep(self, obs):    
        super(ResourceMngrSubAgent, self).FirstStep()  
      
        self.numScvAll = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]
        self.rallyPointSet = False

        self.current_state = np.zeros(RESOURCE_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(RESOURCE_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(RESOURCE_STATE.SIZE, dtype=np.int32, order='C')

        self.gasGatherTarget = {}
        self.gasGatherTarget[SCV_GROUP_GAS1] = None
        self.gasGatherTarget[SCV_GROUP_GAS2] = None


    def Learn(self, reward, terminal):            
        if self.trainAgent:
            if self.isActionCommitted:
                self.decisionMaker.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(ACTIONS.NUM_ACTIONS):
                    self.decisionMaker.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.decisionMaker.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

    def ValidActions(self):
        valid = [ACTIONS.ID_DO_NOTHING]
        if self.current_scaled_state[RESOURCE_STATE.MINERALS_IDX] >= self.scvMineralPrice:
            valid.append(ACTIONS.ID_CREATE_SCV)
        
        for fromGroup in ALL_SCV_GROUPS:
            if self.current_scaled_state[RESOURCE_STATE.GROUP2IDX[fromGroup]] > 0:
                for toGroup in ALL_SCV_GROUPS:
                    if fromGroup != toGroup:
                        valid.append(ACTIONS.GROUP2ACTION[toGroup, fromGroup])

        return valid

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions()
                if self.trainAgent:
                    targetValues = False
                    exploreProb = self.decisionMaker.ExploreProb()              
                else:
                    targetValues = True
                    exploreProb = self.decisionMaker.TargetExploreProb()   

                if np.random.uniform() > exploreProb:
                    valVec = self.decisionMaker.ActionValuesVec(self.current_scaled_state, targetValues)  
                    random.shuffle(validActions)
                    validVal = valVec[validActions]
                    action = validActions[validVal.argmax()]
                else:
                    action = np.random.choice(validActions) 
            else:
                action = self.decisionMaker.choose_action(self.current_scaled_state)
        else:
            action = self.subAgentPlay

        self.current_action = action
        return action

    def IsDoNothingAction(self, a):
        return a == ACTIONS.ID_DO_NOTHING

    def GetMineralGroup(self):
        return SCV_GROUP_MINERALS

    def CreateState(self, obs):
        for i in range(len(GAS_GROUPS)):
            if self.gasGatherTarget[GAS_GROUPS[i]] == None:
                if len(self.sharedData.buildingCompleted[Terran.Refinery]) > i:
                    self.gasGatherTarget[GAS_GROUPS[i]] = self.sharedData.buildingCompleted[Terran.Refinery][i].m_screenLocation
                    
        unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]

        self.current_state[RESOURCE_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[RESOURCE_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        self.current_state[RESOURCE_STATE.MINERALS_FIELDS_COUNT] = self.CountMineralFields(unitType)
        self.current_state[RESOURCE_STATE.GAS_REFINERY_COUNT] = len(self.sharedData.buildingCompleted[Terran.Refinery])

        for group, stateIdx in RESOURCE_STATE.GROUP2IDX.items():
            size = obs.observation['control_groups'][group][SC2_Params.NUM_UNITS_CONTROL_GROUP]
            prevSize = self.current_state[stateIdx]
            if group in GAS_GROUPS and size + 1 <= prevSize:
                self.current_state[stateIdx] = max(size, prevSize)
            else:
                self.current_state[stateIdx] = size

        numScv = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]        
        for i in range(self.numScvAll, numScv):
            self.sharedData.scvBuildingQ.pop(0)
        
        self.numScvAll = numScv

        self.current_state[RESOURCE_STATE.SCV_BUILDING_QUEUE] = len(self.sharedData.scvBuildingQ)

        self.ScaleState()
    
    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[RESOURCE_STATE.MINERALS_IDX] = int(self.current_scaled_state[RESOURCE_STATE.MINERALS_IDX] / RESOURCE_STATE.MINERALS_BUCKETING) * RESOURCE_STATE.MINERALS_BUCKETING
        self.current_scaled_state[RESOURCE_STATE.MINERALS_IDX] = min(RESOURCE_STATE.MINERALS_MAX, self.current_scaled_state[RESOURCE_STATE.MINERALS_IDX])
        self.current_scaled_state[RESOURCE_STATE.GAS_IDX] = int(self.current_scaled_state[RESOURCE_STATE.GAS_IDX] / RESOURCE_STATE.GAS_BUCKETING) * RESOURCE_STATE.GAS_BUCKETING
        self.current_scaled_state[RESOURCE_STATE.GAS_IDX] = min(RESOURCE_STATE.GAS_MAX, self.current_scaled_state[RESOURCE_STATE.GAS_IDX])

    def CountMineralFields(self, unitType):
        count = 0
        for res in SC2_Params.NEUTRAL_MINERAL_FIELD:
            count += np.count_nonzero(unitType == res)

        return count

    def Action2Str(self, a):
        return ACTIONS.ACTION2STR[a]

    def Action2SC2Action(self, obs, action, moveNum):
        if action == ACTIONS.ID_CREATE_SCV:
            return self.ActionCreateScv(obs, moveNum)
        else:
            toGroup, fromGroup = (ACTIONS.GROUP2ACTION == action).nonzero()
            if len(toGroup) > 0:
                return self.ActionAdd2Group(obs, moveNum, toGroup[0], fromGroup[0])

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionAdd2Group(self, obs, moveNum, toGroup, fromGroup):
        if moveNum == 0:
            return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, [fromGroup]]), False

        elif moveNum == 1:
            unitSelected = obs.observation['feature_screen'][SC2_Params.SELECTED_IN_SCREEN]
            unit_y, unit_x = SelectUnitValidPoints(unitSelected != 0) 
            if len(unit_y) > 0:
                target = [unit_x[0], unit_y[0]]
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, target]), False

        elif moveNum == 2:
            scvStatus = list(obs.observation['single_select'])
            if len(scvStatus) == 1:  
                if toGroup == SCV_GROUP_GAS1:
                    self.sharedData.scvGasGroups[0] = SCV_GROUP_GAS1
                elif toGroup == SCV_GROUP_GAS2:
                    self.sharedData.scvGasGroups[1] = SCV_GROUP_GAS2

                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_APPEND_AND_STEAL, [ALL_SCV_GROUPS[toGroup]]]), False

        elif moveNum == 3:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.GroupTargetResources(toGroup, obs.observation['feature_screen'][SC2_Params.UNIT_TYPE])
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionCreateScv(self, obs, moveNum):
        if moveNum == 0:
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            target = SelectBuildingValidPoint(unitType, Terran.CommandCenter)
            if target[0] >= 0:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), False
        
        elif moveNum == 1:
            if SC2_Actions.TRAIN_SCV in obs.observation['available_actions']:
                terminal = self.rallyPointSet
                self.sharedData.scvBuildingQ.append(ScvCmd())
                return actions.FunctionCall(SC2_Actions.TRAIN_SCV, [SC2_Params.QUEUED]), terminal

        if moveNum == 2:
            if SC2_Actions.RALLY_SCV in obs.observation['available_actions']:
                coord = self.rallyCoordScv
                self.rallyPointSet = True
                return actions.FunctionCall(SC2_Actions.RALLY_SCV, [SC2_Params.NOT_QUEUED, coord]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True 

    def GroupTargetResources(self, group, unitType):
        if group == SCV_GROUP_MINERALS or self.gasGatherTarget[group] == None:
             target = GatherResource(unitType, SC2_Params.NEUTRAL_MINERAL_FIELD)
             if target[0] < 0:
                shuffleKeys = random.shuffle(list(self.gasGatherTarget.keys()))
                for key in shuffleKeys:
                    if self.gasGatherTarget[key] != None:
                        return self.gasGatherTarget[key]
        else:
            return self.gasGatherTarget[group]
        
        return [-1,-1]





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

# # for independent run
# class Gather(BaseAgent):
#     def __init__(self, runArg = None, decisionMaker = None, isMultiThreaded = False):
#         super(Gather, self).__init__(STATE.SIZE)

#         if runArg == None:
#             runTypeArg = list(ALL_TYPES.intersection(sys.argv))
#             runArg = runTypeArg.pop()
#         self.agent = ResourceMngrSubAgent(runArg, decisionMaker, isMultiThreaded)

#         self.maxNumRef = 2
#         self.refineryMineralsCost = 75
    
#     def GetDecisionMaker(self):
#         return self.agent.GetDecisionMaker()

#     def step(self, obs):
#         super(Gather, self).step(obs)

#         if obs.first():
#             self.FirstStep(obs)

#         a = self.agent.step(obs,self.moveNum, self.sharedData)

#         self.currMin = obs.observation['player'][SC2_Params.MINERALS]

#         if not self.startMineMinerals:
#             sc2Action, terminal = self.StartMine(obs, self.moveNum)
#         elif self.refBuildCmd + self.refBuildComplete != self.maxNumRef and self.currMin >= self.refineryMineralsCost:
#             sc2Action, terminal = self.BuildRefinery(obs, self.moveNum)
#         else:
#             monitor = False
            
#             if self.refBuildComplete != self.maxNumRef and self.refBuildCmd > 0:
#                 idx = np.random.randint(0,10)
#                 if idx > 6:
#                     sc2Action, terminal = self.MonitorRefBuild(obs, self.moveNum)
#                     monitor = True
            
#             if not monitor:
#                 sc2Action, terminal = self.agent.Action2SC2Action(obs, a, self.moveNum)
                

#         self.moveNum = 0 if terminal else self.moveNum + 1
#         return sc2Action

#     def FirstStep(self, obs):
#         self.moveNum = 0
#         self.startMineMinerals = False

#         self.refBuildCmd = 0
#         self.refBuildComplete = 0

#         self.sharedData = SharedDataGather()
#         self.sharedData.scvIdle = obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT]

#     def StartMine(self, obs, moveNum):
#         if moveNum == 0:
#             return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_ALL]), False
#         elif moveNum == 1:
#             numScv = len(obs.observation['multi_select'])
#             if numScv == 0:
#                 numScv = len(obs.observation['single_select'])
            
#             if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
#                 target = self.agent.GatherResource(SC2_Params.NEUTRAL_MINERAL_FIELD)
#                 if target[0] >= 0:

#                     return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), False
        
#         elif moveNum == 2:
#             if obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT] == 0:
#                 self.sharedData.scvIdle = 0
#                 self.sharedData.scvMinerals = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]
#                 self.startMineMinerals = True

#         return SC2_Actions.DO_NOTHING_SC2_ACTION, True

#     def BuildRefinery(self, obs, moveNum):
#         if moveNum == 0:
#             if obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT] > 0:
#                 return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_SINGLE]), False

#             target = self.agent.FindClosestSCVFromRes(SC2_Params.NEUTRAL_MINERAL_FIELD)
                
#             if target[0] >= 0:      
#                 return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), False
        
#         elif moveNum == 1:
#             finishedAction = False
#             sc2Action = TerranUnit.BUILDING_SPEC[Terran.Refinery].sc2Action

#             if sc2Action in obs.observation['available_actions']:
#                 self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
#                 coord = GetLocationForBuilding(obs, Terran.Refinery)

#                 if coord[SC2_Params.Y_IDX] >= 0:
#                     self.refBuildCmd += 1
#                     self.sharedData.scvMinerals -= 1
#                     return actions.FunctionCall(sc2Action, [SC2_Params.NOT_QUEUED, SwapPnt(coord)]), finishedAction

#         return SC2_Actions.DO_NOTHING_SC2_ACTION, True

#     def MonitorRefBuild(self, obs, moveNum):
#         if moveNum == 0:
#             target = SelectBuildingValidPoint(obs.observation['feature_screen'][SC2_Params.UNIT_TYPE], Terran.Refinery)
#             if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
#                 return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), False

#         elif moveNum == 1:
#             refNum = self.RefineryNum(obs)
#             if self.refBuildComplete < refNum:
#                 cmdEnded = refNum - self.refBuildComplete
#                 self.refBuildCmd -= cmdEnded
#                 self.refBuildComplete = refNum
#                 self.sharedData.activeRefinery = refNum
#                 self.sharedData.scvGas += cmdEnded

#         return SC2_Actions.DO_NOTHING_SC2_ACTION, True

#     def RefineryNum(self, obs):
#         buildingStatus = obs.observation['multi_select']
#         if len(buildingStatus) == 0:
#             buildingStatus = obs.observation['single_select']
#             if len(buildingStatus) == 0:
#                 return self.refBuildComplete

#         if buildingStatus[0][SC2_Params.UNIT_TYPE_IDX] != Terran.Refinery:
#             return self.refBuildComplete

#         numComplete = 0
#         for stat in buildingStatus[:]:
#             if stat[SC2_Params.COMPLETION_RATIO_IDX] == 0:
#                 numComplete += 1
        
#         return numComplete