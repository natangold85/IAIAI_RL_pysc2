# build base sub agent
import random
import math
import os.path
import time
import sys

import numpy as np
import pandas as pd

from pysc2.lib import actions

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
from utils import GetScreenCorners
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import GetLocationForBuilding
from utils import GetLocationForBuildingAddition
from utils import IsolateArea
from utils import SelectBuildingValidPoint


AGENT_DIR = "BuildBase/"
if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)
AGENT_NAME = "builder"

# possible types of decision maker

QTABLE = 'q'
DQN = 'dqn'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding' 

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, DQN_EMBEDDING_LOCATIONS])

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "hist"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'


class ACTIONS:
    ID_DO_NOTHING = 0
    ID_BUILD_SUPPLY_DEPOT = 1
    ID_BUILD_REFINERY = 2
    ID_BUILD_BARRACKS = 3
    ID_BUILD_FACTORY = 4
    ID_BUILD_BARRACKS_REACTOR = 5
    ID_BUILD_FACTORY_TECHLAB = 6
    NUM_ACTIONS = 7
    
    ACTION2STR = ["DoNothing" , "BuildSupplyDepot", "BuildOilRefinery", "BuildBarracks", "BuildFactory", "BuildBarrackReactor", "BuildFactoryTechLab"]

    BUILDING_2_ACTION_TRANSITION = {}
    BUILDING_2_ACTION_TRANSITION[TerranUnit.SUPPLY_DEPOT] = ID_BUILD_SUPPLY_DEPOT
    BUILDING_2_ACTION_TRANSITION[TerranUnit.OIL_REFINERY] = ID_BUILD_REFINERY
    BUILDING_2_ACTION_TRANSITION[TerranUnit.BARRACKS] = ID_BUILD_BARRACKS
    BUILDING_2_ACTION_TRANSITION[TerranUnit.FACTORY] = ID_BUILD_FACTORY

    ACTION_2_BUILDING_TRANSITION = {}
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_SUPPLY_DEPOT] = TerranUnit.SUPPLY_DEPOT
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_REFINERY] = TerranUnit.OIL_REFINERY
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_BARRACKS] = TerranUnit.BARRACKS
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_FACTORY] = TerranUnit.FACTORY
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_BARRACKS_REACTOR] = [TerranUnit.BARRACKS, TerranUnit.REACTOR]
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_FACTORY_TECHLAB] = [TerranUnit.FACTORY, TerranUnit.TECHLAB]

class STATE:
    # state details
    MINERALS_MAX = 500
    GAS_MAX = 300
    MINERALS_BUCKETING = 50
    GAS_BUCKETING = 50

    COMMAND_CENTER_IDX = 0
    MINERALS_IDX = 1
    GAS_IDX = 2
    SUPPLY_DEPOT_IDX = 3
    REFINERY_IDX = 4
    BARRACKS_IDX = 5
    FACTORY_IDX = 6
    REACTORS_IDX = 7
    TECHLAB_IDX = 8

    IN_PROGRESS_SUPPLY_DEPOT_IDX = 9
    IN_PROGRESS_REFINERY_IDX = 10
    IN_PROGRESS_BARRACKS_IDX = 11
    IN_PROGRESS_FACTORY_IDX = 12
    IN_PROGRESS_RECTORS_IDX = 13
    IN_PROGRESS_TECHLAB_IDX = 14    
    
    SIZE = 15

    BUILDING_2_STATE_TRANSITION = {}
    BUILDING_2_STATE_TRANSITION[TerranUnit.COMMANDCENTER] = [COMMAND_CENTER_IDX, -1]
    BUILDING_2_STATE_TRANSITION[TerranUnit.SUPPLY_DEPOT] = [SUPPLY_DEPOT_IDX, IN_PROGRESS_SUPPLY_DEPOT_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.OIL_REFINERY] = [REFINERY_IDX, IN_PROGRESS_REFINERY_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.BARRACKS] = [BARRACKS_IDX, IN_PROGRESS_BARRACKS_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.FACTORY] = [FACTORY_IDX, IN_PROGRESS_FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.REACTOR] = [REACTORS_IDX, IN_PROGRESS_RECTORS_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.TECHLAB] = [TECHLAB_IDX, IN_PROGRESS_TECHLAB_IDX]

class ActionRequirement:
    def __init__(self, mineralsPrice = 0, gasPrice = 0, buildingDependency = STATE.MINERALS_IDX):
        self.mineralsPrice = mineralsPrice
        self.gasPrice = gasPrice
        self.buildingDependency = buildingDependency



# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[QTABLE][DIRECTORY] = "buildBase_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "build_dqn"
RUN_TYPES[DQN][DIRECTORY] = "buildBase_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

class SharedDataBuild(EmptySharedData):
    def __init__(self):
        super(SharedDataBuild, self).__init__()

        # building commands
        self.buildCommands = {}
        self.buildingCount = {}
        for key in STATE.BUILDING_2_STATE_TRANSITION.keys():
            self.buildingCount[key] = 0
            self.buildCommands[key] = []

        self.commandCenterLoc = []

class BuildingCmd:
    def __init__(self, inProgress = False):
        self.m_inProgress = inProgress
        self.m_stepsCounter = 0

class BuildBaseSubAgent:
    def __init__(self, runArg = None, decisionMaker = None, isMultiThreaded = False, playList = None, trainList = None):        
        
        self.discountFactor = 0.95
        
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent =AGENT_NAME in trainList
        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(runArg, isMultiThreaded)

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        # states and action:
        self.terminalState = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        # model params
        self.unit_type = None

        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]

        self.actionsRequirement = {}
        self.actionsRequirement[ACTIONS.ID_BUILD_SUPPLY_DEPOT] = ActionRequirement(100)
        self.actionsRequirement[ACTIONS.ID_BUILD_REFINERY] = ActionRequirement(75)
        self.actionsRequirement[ACTIONS.ID_BUILD_BARRACKS] = ActionRequirement(150,0,STATE.SUPPLY_DEPOT_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_FACTORY] = ActionRequirement(150,100,STATE.BARRACKS_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_BARRACKS_REACTOR] = ActionRequirement(50,50,STATE.BARRACKS_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_FACTORY_TECHLAB] = ActionRequirement(50,25,STATE.FACTORY_IDX)

        self.maxNumOilRefinery = 2

    def CreateDecisionMaker(self, runArg, isMultiThreaded):
        if runArg == None:
            runTypeArg = list(ALL_TYPES.intersection(sys.argv))
            runArg = runTypeArg.pop()    
        runType = RUN_TYPES[runArg]

        decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                        resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=AGENT_DIR + runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1
        
    def step(self, obs, sharedData = None, moveNum = None):  

        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]
        
        if obs.first():
            self.FirstStep(obs)
        
        if sharedData != None:
            self.sharedData = sharedData

        if moveNum == 0: 
            self.CreateState(obs)
            self.Learn()
            self.current_action = self.ChooseAction()

        self.numSteps += 1

        return self.current_action

    def Action2SC2Action(self, obs, a, moveNum):
        self.isActionCommitted = True

        if a == ACTIONS.ID_DO_NOTHING:
            return SC2_Actions.DO_NOTHING_SC2_ACTION, True
        elif a < ACTIONS.ID_BUILD_BARRACKS_REACTOR:
            return self.BuildAction(obs, moveNum)
        elif a < ACTIONS.NUM_ACTIONS:
            return self.BuildAdditionAction(obs, moveNum)

    def FirstStep(self, obs):
        self.numSteps = 0

        player_y, player_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()

        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        self.current_action = None
        self.previous_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        commandCenterLoc_y, commandCenterLoc_x = (self.unit_type == TerranUnit.COMMANDCENTER).nonzero()

        middleCC = FindMiddle(commandCenterLoc_y, commandCenterLoc_x)
        miniMapLoc = Scale2MiniMap(middleCC, self.cameraCornerNorthWest , self.cameraCornerSouthEast)
        
        self.sharedData = SharedDataBuild()
        self.sharedData.buildingCount[TerranUnit.COMMANDCENTER] += 1        
        self.sharedData.CommandCenterLoc = [miniMapLoc]


        self.isActionCommitted = False
        self.stepActionCommmitted = 0
        self.lastActionCommittedAction = None
        self.lastActionCommittedState = None
        self.lastActionCommittedNextState = None

    def LastStep(self, obs, reward):
        if self.lastActionCommittedAction is not None:
            self.decisionMaker.learn(self.lastActionCommittedState.copy(), self.current_action, reward, self.lastActionCommittedNextState.copy(), True)

            score = obs.observation["score_cumulative"][0]
            self.decisionMaker.end_run(reward, score, self.numSteps)


    def Learn(self, reward = 0):

        if self.trainAgent:
            if self.isActionCommitted:
                self.decisionMaker.learn(self.previous_scaled_state, self.current_action, reward, self.current_scaled_state)
            elif reward > 0 and self.lastActionCommittedAction != None:
                discountedReward = reward * pow(self.discountFactor, self.numSteps - self.stepActionCommmitted)
                self.decisionMaker.learn(self.lastActionCommittedState, self.lastActionCommittedAction, discountedReward, self.lastActionCommittedNextState)
        
        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False
        self.lastActionCommittedAction = None

    def IsDoNothingAction(self, a):
        return a == ACTIONS.ID_DO_NOTHING

    def BuildAction(self, obs, moveNum):
        if moveNum == 0:
            finishedAction = False
            # select scv
            unit_y, unit_x = (self.unit_type == TerranUnit.SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, target]), finishedAction
        
        elif moveNum == 1:
            finishedAction = False
            buildingType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action]
            sc2Action = TerranUnit.UNIT_2_SC2ACTIONS[buildingType]

            # preform build action. if action not available go back to harvest
            if sc2Action in obs.observation['available_actions']:
                coord = GetLocationForBuilding(obs, self.cameraCornerNorthWest, self.cameraCornerSouthEast, buildingType)

                if coord[SC2_Params.Y_IDX] >= 0:
                    self.sharedData.buildCommands[buildingType].append(BuildingCmd())
                    return actions.FunctionCall(sc2Action, [SC2_Params.NOT_QUEUED, SwapPnt(coord)]), finishedAction
            moveNum += 1

        if moveNum == 2:
            finishedAction = True
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.GatherHarvest()
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), finishedAction
     
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def BuildAdditionAction(self, obs, moveNum):
        buildingType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action][0]
        additionType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action][1]

        if moveNum == 0:
            # select building without addition
            target = SelectBuildingValidPoint(self.unit_type, buildingType)
            if target[0] >= 0:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), False
            
        elif moveNum == 1:
            action = TerranUnit.UNIT_2_SC2ACTIONS[additionType]
            if action in obs.observation['available_actions']:
                coord = GetLocationForBuildingAddition(obs, buildingType, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                if coord[0] >= 0:
                    
                    numBuildingsCompleted = self.current_state[STATE.BUILDING_2_STATE_TRANSITION[buildingType][0]]
                    numAdditionAll = sum(self.current_state[STATE.BUILDING_2_STATE_TRANSITION[additionType]])
                    if numBuildingsCompleted > numAdditionAll:
                        self.sharedData.buildCommands[additionType].append(BuildingCmd())
                    
                    return actions.FunctionCall(action, [SC2_Params.QUEUED, SwapPnt(coord)]) , True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True    

    def CreateState(self, obs):
        for key, value in STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.sharedData.buildingCount[key]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.sharedData.buildCommands[key])
    
        self.current_state[STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        self.ScaleState()

        if self.isActionCommitted:
            self.stepActionCommmitted = self.numSteps
            self.lastActionCommittedAction = self.current_action
            self.lastActionCommittedState = self.previous_scaled_state
            self.lastActionCommittedNextState = self.current_scaled_state
   
    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[STATE.MINERALS_IDX] = int(self.current_scaled_state[STATE.MINERALS_IDX] / STATE.MINERALS_BUCKETING) * STATE.MINERALS_BUCKETING
        self.current_scaled_state[STATE.MINERALS_IDX] = min(STATE.MINERALS_MAX, self.current_scaled_state[STATE.MINERALS_IDX])
        self.current_scaled_state[STATE.GAS_IDX] = int(self.current_scaled_state[STATE.GAS_IDX] / STATE.GAS_BUCKETING) * STATE.GAS_BUCKETING
        self.current_scaled_state[STATE.GAS_IDX] = min(STATE.GAS_MAX, self.current_scaled_state[STATE.GAS_IDX])

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions()
                if self.trainAgent:
                    targetValues = False
                    exploreProb = self.decisionMaker.ExploreProb()              
                else:
                    targetValues = True
                    exploreProb = 0   

                if np.random.uniform() > exploreProb:
                    valVec = self.decisionMaker.ActionValuesVec(self.current_state, targetValues)
                    random.shuffle(validActions)
                    validVal = valVec[validActions]
                    action = validActions[validVal.argmax()]
                else:
                    action = np.random.choice(validActions) 
            else:
                action = self.decisionMaker.choose_action(self.current_state)
        else:
            action = self.subAgentPlay
        
        return action

    def ValidActions(self):
        valid = [ACTIONS.ID_DO_NOTHING]
        for key, requirement in self.actionsRequirement.items():
            if self.ValidSingleAction(requirement):
                valid.append(key)
        # special condition for oil refinery:
        if self.current_scaled_state[STATE.REFINERY_IDX] + self.current_scaled_state[STATE.IN_PROGRESS_REFINERY_IDX] >= self.maxNumOilRefinery and ACTIONS.ID_BUILD_REFINERY in valid:
            valid.remove(ACTIONS.ID_BUILD_REFINERY)

        return valid

    def ValidSingleAction(self, requirement):
        hasMinerals = self.current_scaled_state[STATE.MINERALS_IDX] >= requirement.mineralsPrice
        hasGas = self.current_scaled_state[STATE.GAS_IDX] >= requirement.gasPrice
        otherReq = self.current_scaled_state[requirement.buildingDependency] > 0
        return hasMinerals & hasGas & otherReq
    
    def GatherHarvest(self):
        if random.randint(0, 4) < 4:
            resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD
        else:
            resourceList = [TerranUnit.OIL_REFINERY]

        unit_y = []
        unit_x = []
        for val in resourceList[:]:
            p_y, p_x = (self.unit_type == val).nonzero()
            unit_y += list(p_y)
            unit_x += list(p_x)

        if len(unit_y) > 0:
            i = random.randint(0, len(unit_y) - 1)
            return [unit_y[i], unit_x[i]]
        
        return [-1,-1]

    def Action2Str(self, a):
        return ACTIONS.ACTION2STR[a]

    def HasAddition(self, pnt_y, pnt_x, additionType):
        additionMat = self.unit_type == additionType
        
        for i in range(0, len(pnt_y)):
            nearX = pnt_x[i] + 1
            if nearX < SC2_Params.SCREEN_SIZE and additionMat[pnt_y[i]][nearX]:
                return True

        return False

