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
from algo_decisionMaker import BaseNaiveDecisionMaker
from algo_decisionMaker import DecisionMakerExperienceReplay
from algo_decisionMaker import UserPlay

from utils_results import ResultFile
from utils_results import PlotResults

# params
from algo_dqn import DQN_PARAMS
from algo_dqn import DQN_EMBEDDING_PARAMS
from algo_qtable import QTableParams
from algo_qtable import QTableParamsExplorationDecay

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
HISTORY = "history"
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
    
    SCV_NUM = 4
    SCV_GROUP_MINERALS_IDX = 5
    SCV_GROUP_GAS1_IDX = 6
    SCV_GROUP_GAS2_IDX = 7
    SCV_BUILDING_QUEUE = 8
    
    SIZE = 9

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
    def __init__(self, resultFName = None, directory = None, numTrials2Save = 20):
        super(NaiveDecisionMakerResource, self).__init__(numTrials2Save=numTrials2Save, resultFName=resultFName, directory=directory, agentName=AGENT_NAME)
        self.desiredGroupCnt = {}
        self.desiredGroupCnt[SCV_GROUP_MINERALS] = 8
        self.desiredGroupCnt[SCV_GROUP_GAS1] = 3
        self.desiredGroupCnt[SCV_GROUP_GAS2] = 3

        self.desiredScvSize = 8 + 3+ 3
        self.maxQSize = 5

    def choose_action(self, state, validActions, targetValues=False):
        hasRes = state[RESOURCE_STATE.MINERALS_IDX] > RESOURCE_STATE.SCV_MINERALS_PRICE
        minSize = state[RESOURCE_STATE.SCV_GROUP_MINERALS_IDX]
        qSize = state[RESOURCE_STATE.SCV_BUILDING_QUEUE]

        gas1Size = state[RESOURCE_STATE.SCV_GROUP_GAS1_IDX]
        gas2Size = state[RESOURCE_STATE.SCV_GROUP_GAS2_IDX]
        
        sumScv = max(minSize + gas1Size + gas2Size + qSize, state[RESOURCE_STATE.SCV_NUM])
        action = ACTIONS.ID_DO_NOTHING
        if minSize > self.desiredGroupCnt[SCV_GROUP_MINERALS]:
            if gas1Size < self.desiredGroupCnt[SCV_GROUP_GAS1]:
                action = ACTIONS.ID_ADD2_GAS1_FROM_MINERALS
            else:
                action = ACTIONS.ID_ADD2_GAS2_FROM_MINERALS
        elif sumScv < self.desiredScvSize and hasRes:
            action = ACTIONS.ID_CREATE_SCV
        
        return action if action in validActions else ACTIONS.ID_DO_NOTHING

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

    def ActionsValues(self, state, validActions, target = True):    
        vals = np.zeros(ACTIONS.NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals



class ResourceMngrSubAgent(BaseAgent):
    def __init__(self, sharedData, configDict, decisionMaker, isMultiThreaded, playList, trainList, testList, dmCopy=None):
        super(ResourceMngrSubAgent, self).__init__(RESOURCE_STATE.SIZE)

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(configDict, isMultiThreaded)

        self.history = self.decisionMaker.AddHistory()

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

    def CreateDecisionMaker(self, configDict, isMultiThreaded, dmCopy=None):
        decisionMaker = NaiveDecisionMakerResource()
        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
            
        return None

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
            reward = reward if not terminal else self.NormalizeReward(reward)

            if self.isActionCommitted:
                self.history.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(ACTIONS.NUM_ACTIONS):
                    self.history.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.history.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent or self.testAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

    def ValidActions(self, state):
        valid = [ACTIONS.ID_DO_NOTHING]
        if state[RESOURCE_STATE.MINERALS_IDX] >= self.scvMineralPrice:
            valid.append(ACTIONS.ID_CREATE_SCV)
        
        for fromGroup in ALL_SCV_GROUPS:
            if state[RESOURCE_STATE.GROUP2IDX[fromGroup]] > 0:
                for toGroup in ALL_SCV_GROUPS:
                    if fromGroup != toGroup:
                        valid.append(ACTIONS.GROUP2ACTION[toGroup, fromGroup])

        return valid

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                # todo: create valid actions for agent
                validActions = self.ValidActions(self.current_scaled_state)
            else: 
                validActions = list(range(ACTIONS.NUM_ACTIONS))
 
            targetValues = False if self.trainAgent else True
            return self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)
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


        self.current_state[RESOURCE_STATE.SCV_NUM] = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]
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

    def Action2Str(self, a, onlyAgent=False):
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
