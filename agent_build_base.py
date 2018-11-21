# build base sub agent
import random
import math
import os.path
import time
import sys
import datetime
from multiprocessing import Lock

import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils import BaseAgent

#decision makers
from algo_decisionMaker import BaseDecisionMaker
from algo_decisionMaker import DecisionMakerExperienceReplay
from algo_decisionMaker import UserPlay
from algo_decisionMaker import BaseNaiveDecisionMaker

from utils_results import ResultFile
from utils_results import PlotResults

# params
from algo_dqn import DQN_PARAMS
from algo_dqn import DQN_EMBEDDING_PARAMS
from algo_dqn import DQN_PARAMS_WITH_DEFAULT_DM

from algo_a2c import A2C_PARAMS

from algo_qtable import QTableParams
from algo_qtable import QTableParamsExplorationDecay

# utils functions
from utils import EmptySharedData
from utils import SwapPnt
from utils import GetScreenCorners
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import GetLocationForBuilding
from utils import IsolateArea
from utils import SelectBuildingValidPoint
from utils import SelectUnitValidPoints
from utils import GatherResource

AGENT_DIR = "BuildBase/"
AGENT_NAME = "build_base"

# possible types of decision maker

QTABLE = 'q'
DQN = 'dqn'
DQN2L = 'dqn_2l'
DQN2L_DFLT = 'dqn_2l_dflt'
A2C = "A2C"
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding' 
NAIVE = "naive"
USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, DQN_EMBEDDING_LOCATIONS, NAIVE])

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "history"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

ADDITION_TYPES = [Terran.BarracksReactor, Terran.FactoryTechLab]

ADDITION_2_BUILDING = {}
ADDITION_2_BUILDING[Terran.BarracksReactor] = Terran.Barracks
ADDITION_2_BUILDING[Terran.FactoryTechLab] = Terran.Factory

QUICK_ADDITION_ACTION = {}
QUICK_ADDITION_ACTION[Terran.BarracksReactor] = SC2_Actions.BUILD_REACTOR_QUICK
QUICK_ADDITION_ACTION[Terran.FactoryTechLab] = SC2_Actions.BUILD_TECHLAB_QUICK

ADDITION_ACTION = {}
ADDITION_ACTION[Terran.BarracksReactor] = SC2_Actions.BUILD_REACTOR
ADDITION_ACTION[Terran.FactoryTechLab] = SC2_Actions.BUILD_TECHLAB

# Model Params
NUM_TRIALS_2_LEARN = 20
NUM_TRIALS_4_CMP = 200

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
    BUILDING_2_ACTION_TRANSITION[Terran.SupplyDepot] = ID_BUILD_SUPPLY_DEPOT
    BUILDING_2_ACTION_TRANSITION[Terran.Refinery] = ID_BUILD_REFINERY
    BUILDING_2_ACTION_TRANSITION[Terran.Barracks] = ID_BUILD_BARRACKS
    BUILDING_2_ACTION_TRANSITION[Terran.Factory] = ID_BUILD_FACTORY
    BUILDING_2_ACTION_TRANSITION[Terran.BarracksReactor] = ID_BUILD_BARRACKS_REACTOR
    BUILDING_2_ACTION_TRANSITION[Terran.FactoryTechLab] = ID_BUILD_FACTORY_TECHLAB

    ACTION_2_BUILDING_TRANSITION = {}
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_SUPPLY_DEPOT] = Terran.SupplyDepot
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_REFINERY] = Terran.Refinery
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_BARRACKS] = Terran.Barracks
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_FACTORY] = Terran.Factory
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_BARRACKS_REACTOR] = [Terran.Barracks, Terran.BarracksReactor]
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_FACTORY_TECHLAB] = [Terran.Factory, Terran.FactoryTechLab]

class BUILD_STATE:
    # state details
    MINERALS_MAX = 500
    GAS_MAX = 300
    MINERALS_BUCKETING = 50
    GAS_BUCKETING = 50

    SUPPLY_BUCKETING = 8
    SUPPLY_LEFT_MAX = 20

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
    IN_PROGRESS_REACTORS_IDX = 13
    IN_PROGRESS_TECHLAB_IDX = 14    
    
    SUPPLY_LEFT_IDX = 15

    TIME_LINE_IDX = 16

    SIZE = 17

    BUILDING_2_STATE_TRANSITION = {}
    BUILDING_2_STATE_TRANSITION[Terran.CommandCenter] = [COMMAND_CENTER_IDX, -1]
    BUILDING_2_STATE_TRANSITION[Terran.SupplyDepot] = [SUPPLY_DEPOT_IDX, IN_PROGRESS_SUPPLY_DEPOT_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Refinery] = [REFINERY_IDX, IN_PROGRESS_REFINERY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Barracks] = [BARRACKS_IDX, IN_PROGRESS_BARRACKS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Factory] = [FACTORY_IDX, IN_PROGRESS_FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.BarracksReactor] = [REACTORS_IDX, IN_PROGRESS_REACTORS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.FactoryTechLab] = [TECHLAB_IDX, IN_PROGRESS_TECHLAB_IDX]

    BUILDING_2_ADDITION = {}
    BUILDING_2_ADDITION[Terran.Barracks] = Terran.BarracksReactor
    BUILDING_2_ADDITION[Terran.Factory] = Terran.FactoryTechLab

    IDX2STR = {}

    IDX2STR[COMMAND_CENTER_IDX] = "CC"
    IDX2STR[MINERALS_IDX] = "MIN"
    IDX2STR[GAS_IDX] = "GAS"
    IDX2STR[SUPPLY_DEPOT_IDX] = "SD"
    IDX2STR[REFINERY_IDX] = "REF"
    IDX2STR[BARRACKS_IDX] = "BA"
    IDX2STR[FACTORY_IDX] = "FA"
    IDX2STR[REACTORS_IDX] =  "REA"
    IDX2STR[TECHLAB_IDX] = "TECH"

    IDX2STR[IN_PROGRESS_SUPPLY_DEPOT_IDX] = "SD_B"
    IDX2STR[IN_PROGRESS_REFINERY_IDX] = "REF_B"
    IDX2STR[IN_PROGRESS_BARRACKS_IDX] = "BA_B"
    IDX2STR[IN_PROGRESS_FACTORY_IDX] = "FA_B"
    IDX2STR[IN_PROGRESS_REACTORS_IDX] = "REA_B"
    IDX2STR[IN_PROGRESS_TECHLAB_IDX] = "TECH_B"
    IDX2STR[SUPPLY_LEFT_IDX] = "SupplyLeft"
    IDX2STR[TIME_LINE_IDX] = "TimeLine"

class ActionRequirement:
    def __init__(self, mineralsPrice = 0, gasPrice = 0, buildingDependency = BUILD_STATE.MINERALS_IDX):
        self.mineralsPrice = mineralsPrice
        self.gasPrice = gasPrice
        self.buildingDependency = buildingDependency



# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(BUILD_STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[QTABLE][DIRECTORY] = "buildBase_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(BUILD_STATE.SIZE, ACTIONS.NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "build_dqn"
RUN_TYPES[DQN][DIRECTORY] = "buildBase_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

RUN_TYPES[DQN2L] = {}
RUN_TYPES[DQN2L][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN2L][PARAMS] = DQN_PARAMS(BUILD_STATE.SIZE, ACTIONS.NUM_ACTIONS, layersNum=2, numTrials2CmpResults=NUM_TRIALS_4_CMP, descendingExploration=False)
RUN_TYPES[DQN2L][DECISION_MAKER_NAME] = "build_dqn2l"
RUN_TYPES[DQN2L][DIRECTORY] = "buildBase_dqn2l"
RUN_TYPES[DQN2L][HISTORY] = "replayHistory"
RUN_TYPES[DQN2L][RESULTS] = "result"

RUN_TYPES[DQN2L_DFLT] = {}
RUN_TYPES[DQN2L_DFLT][TYPE] = "DQN_WithTargetAndDefault"
RUN_TYPES[DQN2L_DFLT][PARAMS] = DQN_PARAMS_WITH_DEFAULT_DM(BUILD_STATE.SIZE, ACTIONS.NUM_ACTIONS, layersNum=2, numTrials2CmpResults=NUM_TRIALS_4_CMP, descendingExploration=False)
RUN_TYPES[DQN2L_DFLT][DECISION_MAKER_NAME] = "build_dqn2l_dflt"
RUN_TYPES[DQN2L_DFLT][DIRECTORY] = "buildBase_dqn2l_dflt"
RUN_TYPES[DQN2L_DFLT][HISTORY] = "replayHistory"
RUN_TYPES[DQN2L_DFLT][RESULTS] = "result"

RUN_TYPES[A2C] = {}
RUN_TYPES[A2C][TYPE] = "A2C"
RUN_TYPES[A2C][PARAMS] = A2C_PARAMS(BUILD_STATE.SIZE, ACTIONS.NUM_ACTIONS, numTrials2CmpResults=NUM_TRIALS_4_CMP, outputGraph=True)
RUN_TYPES[A2C][DECISION_MAKER_NAME] = "build_A2C"
RUN_TYPES[A2C][DIRECTORY] = "buildBase_A2C"
RUN_TYPES[A2C][HISTORY] = "replayHistory"
RUN_TYPES[A2C][RESULTS] = "results"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "buildBase_naive"
RUN_TYPES[NAIVE][RESULTS] = "buildBase_result"

BUILDING_NOT_BUILT_THRESHOLD_COUNTER = 40

class SharedDataBuild(EmptySharedData):
    def __init__(self):
        super(SharedDataBuild, self).__init__()
        self.buildCommands = {}
        self.buildingCompleted = {}
        for key in BUILD_STATE.BUILDING_2_STATE_TRANSITION.keys():
            self.buildingCompleted[key] = []
            self.buildCommands[key] = []

        self.commandCenterLoc = []

class Building:
    def __init__(self, screenLocation):
        self.m_screenLocation = screenLocation    
        self.qForProduction = []
        self.m_qHeadSize = 1

    def AddUnit2Q(self, unit):
        if len(self.qForProduction) < self.m_qHeadSize:
            unit.step = 0

        self.qForProduction.append(unit)
    
    def AddStepForUnits(self):
        for unit in self.qForProduction:
            unit.step += 1

    def RemoveUnitFromQ(self, unitType):
        self.qForProduction.sort(key=lambda unit : unit.step, reverse=True)

        for unit in self.qForProduction:
            if unit.unit == unitType:
                self.qForProduction.remove(unit)
                break
        
        for unit in self.qForProduction:
            if unit.step < 0:
                unit.step = 0
                return
                
class BuildingCmd(Building):
    def __init__(self, screenLocation, inProgress = False):
        super(BuildingCmd, self).__init__(screenLocation)
        self.m_inProgress = inProgress
        self.m_stepsCounter = 0

class BuildingCmdAddition(BuildingCmd):  
    def __init__(self, screenLocation, inProgress = False):
        super(BuildingCmdAddition, self).__init__(screenLocation, inProgress)
        self.m_additionCoord = None

def CreateDecisionMakerBuildBase(configDict, isMultiThreaded, dmCopy=None):
    dmCopy = "" if dmCopy==None else "_" + str(dmCopy)

    if configDict[AGENT_NAME] == "none":
        return BaseDecisionMaker(AGENT_NAME), []

    runType = RUN_TYPES[configDict[AGENT_NAME]]
    # create agent dir
    directory = configDict["directory"] + "/" + AGENT_DIR

    if configDict[AGENT_NAME] == "naive":
        decisionMaker = NaiveDecisionMakerBuilder(resultFName=runType[RESULTS], directory=directory + runType[DIRECTORY])
    else:    
        if runType[TYPE] == "DQN_WithTargetAndDefault":
            runType[PARAMS].defaultDecisionMaker = NaiveDecisionMakerBuilder()

        decisionMaker = DecisionMakerExperienceReplay(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], agentName=AGENT_NAME,
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory + runType[DIRECTORY] + dmCopy, isMultiThreaded=isMultiThreaded)

    return decisionMaker, runType

class NaiveDecisionMakerBuilder(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Save=NUM_TRIALS_2_LEARN):
        super(NaiveDecisionMakerBuilder, self).__init__(numTrials2Save=numTrials2Save, resultFName=resultFName, directory=directory, agentName=AGENT_NAME)
        self.SDSupply = 8
        self.CCSupply = 15


    def choose_action(self, state, validActions, targetValues=False):
        action = 0

        numSDAll = state[BUILD_STATE.SUPPLY_DEPOT_IDX] + state[BUILD_STATE.IN_PROGRESS_SUPPLY_DEPOT_IDX]
        numRefAll = state[BUILD_STATE.REFINERY_IDX] + state[BUILD_STATE.IN_PROGRESS_REFINERY_IDX]
        numBaAll = state[BUILD_STATE.BARRACKS_IDX] + state[BUILD_STATE.IN_PROGRESS_BARRACKS_IDX]
        numFaAll = state[BUILD_STATE.FACTORY_IDX] + state[BUILD_STATE.IN_PROGRESS_FACTORY_IDX]
        numReactorsAll = state[BUILD_STATE.REACTORS_IDX] + state[BUILD_STATE.IN_PROGRESS_REACTORS_IDX]
        numTechAll = state[BUILD_STATE.TECHLAB_IDX] + state[BUILD_STATE.IN_PROGRESS_TECHLAB_IDX]
        
        supplyLeft = state[BUILD_STATE.SUPPLY_LEFT_IDX]
        
        if supplyLeft <= 2:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT
        elif numSDAll < 2:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT
        elif numRefAll < 2:
            action = ACTIONS.ID_BUILD_REFINERY
        elif numBaAll < 2:
            action = ACTIONS.ID_BUILD_BARRACKS
        elif numFaAll < 1:
            action = ACTIONS.ID_BUILD_FACTORY
        elif numReactorsAll < 2:
            action = ACTIONS.ID_BUILD_BARRACKS_REACTOR
        elif numSDAll < 6:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT
        elif numFaAll < 2:
            action = ACTIONS.ID_BUILD_FACTORY
        elif numTechAll < 2:
            action = ACTIONS.ID_BUILD_FACTORY_TECHLAB
        elif numBaAll < 4:
            action = ACTIONS.ID_BUILD_BARRACKS
        elif supplyLeft < 6:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT

        return action if action in validActions else ACTIONS.ID_DO_NOTHING

    def ActionsValues(self, state, validActions, target = True):    
        vals = np.zeros(ACTIONS.NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0
        vals[ACTIONS.ID_DO_NOTHING] = 0.1
        return vals


class BuildBaseSubAgent(BaseAgent):
    def __init__(self, sharedData, configDict, decisionMaker, isMultiThreaded, playList, trainList, testList, dmCopy=None):        
        super(BuildBaseSubAgent, self).__init__(BUILD_STATE.SIZE)
        self.discountFactor = 0.95
        
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent =AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList

        self.inTraining = self.trainAgent

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker, _ = CreateDecisionMakerBuildBase(configDict, isMultiThreaded)

        self.history = self.decisionMaker.AddHistory()

        self.sharedData = sharedData

        # states and action:
        self.terminalState = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')

        self.actionsRequirement = {}
        self.actionsRequirement[ACTIONS.ID_BUILD_SUPPLY_DEPOT] = ActionRequirement(100)
        self.actionsRequirement[ACTIONS.ID_BUILD_REFINERY] = ActionRequirement(75)
        self.actionsRequirement[ACTIONS.ID_BUILD_BARRACKS] = ActionRequirement(150,0,BUILD_STATE.SUPPLY_DEPOT_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_FACTORY] = ActionRequirement(150,100,BUILD_STATE.BARRACKS_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_BARRACKS_REACTOR] = ActionRequirement(50,50,BUILD_STATE.BARRACKS_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_FACTORY_TECHLAB] = ActionRequirement(50,25,BUILD_STATE.FACTORY_IDX)

        self.maxNumOilRefinery = 2

        self.notAllowedDirections2CC = [[1, 0], [1, 1], [0, 1]] 

        self.current_state = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')     

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

    def Action2SC2Action(self, obs, a, moveNum):
        if a == ACTIONS.ID_DO_NOTHING:
            return SC2_Actions.DO_NOTHING_SC2_ACTION, True
        elif a < ACTIONS.ID_BUILD_BARRACKS_REACTOR:
            return self.BuildAction(obs, moveNum)
        elif a < ACTIONS.NUM_ACTIONS:
            return self.BuildAdditionAction(obs, moveNum)

    def GetStateVal(self, idx):
        return self.current_state[idx]

    def FirstStep(self, obs):
        super(BuildBaseSubAgent, self).FirstStep()

        self.current_action = None

        self.current_state = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(BUILD_STATE.SIZE, dtype=np.int32, order='C')

        self.lastActionCommittedStep = 0
        self.lastActionCommittedState = None
        self.lastActionCommittedNextState = None

        self.building2AddIdx = -1

    def EndRun(self, reward, score, stepNum):
        saved = False
        if self.trainAgent or self.testAgent:
            saved = self.decisionMaker.end_run(reward, score, stepNum)

        return saved

    def Learn(self, reward, terminal):
        if self.history != None and self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)

            if self.isActionCommitted:
                self.history.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)      
            elif reward > 0:
                if self.lastActionCommitted != None:
                    numSteps = self.sharedData.numAgentStep - self.lastActionCommittedStep
                    discountedReward = reward * pow(self.decisionMaker.DiscountFactor(), numSteps)
                    self.history.learn(self.lastActionCommittedState, self.lastActionCommitted, discountedReward, self.lastActionCommittedNextState, terminal)        
            elif terminal:
                self.history.learn(self.current_scaled_state, ACTIONS.ID_DO_NOTHING, reward, self.terminalState, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False


    def IsDoNothingAction(self, a):
        return a == ACTIONS.ID_DO_NOTHING

    def BuildAction(self, obs, moveNum):
        buildingType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action]

        if moveNum == 0:
            if buildingType == Terran.Refinery:
                numRefinery = self.current_scaled_state[BUILD_STATE.REFINERY_IDX] + self.current_scaled_state[BUILD_STATE.IN_PROGRESS_REFINERY_IDX]               
                group2Select = self.sharedData.scvGasGroups[numRefinery]
            else:
                if self.NumBeforeProgress(buildingType) == 0:
                    group2Select = self.sharedData.scvMineralGroup
                else:
                    group2Select = None


            
            if group2Select != None:
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, [group2Select]]), False
            
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            scv_y, scv_x = SelectUnitValidPoints(unitType == Terran.SCV)
            if len(scv_y) > 0:
                target = [scv_y[0], scv_x[0]]
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), False

        elif moveNum == 1:
            unitSelected = obs.observation['feature_screen'][SC2_Params.SELECTED_IN_SCREEN]
            unit_y, unit_x = SelectUnitValidPoints(unitSelected != 0) 
            if len(unit_y) > 0:
                target = [unit_x[0], unit_y[0]]
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, target]), False
        
        elif moveNum == 2:
            finishedAction = buildingType == Terran.Refinery
            sc2Action = TerranUnit.BUILDING_SPEC[buildingType].sc2Action

            # preform build action. if action not available go back to harvest
            if sc2Action in obs.observation['available_actions']:
                buildingCmdLocations = self.GetBuildingCmdAdditionLocations()
                coord = GetLocationForBuilding(obs, buildingType, self.notAllowedDirections2CC, buildingCmdLocations)
                if coord[SC2_Params.Y_IDX] >= 0:
                    self.sharedData.buildCommands[buildingType].append(BuildingCmd(coord))
                    return actions.FunctionCall(sc2Action, [SC2_Params.NOT_QUEUED, SwapPnt(coord)]), finishedAction
            moveNum += 1

        if moveNum == 3:
            finishedAction = True
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = GatherResource(unitType, SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] < 0:
                    target = GatherResource(unitType, [Terran.Refinery])

                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), finishedAction
     
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def BuildAdditionAction(self, obs, moveNum):
        buildingType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action][0]
        additionType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action][1]

        if moveNum == 0:
            # select building without addition
            self.building2AddIdx = self.SelectBuilding2Add(buildingType)
            if self.building2AddIdx >= 0:
                coord = self.sharedData.buildingCompleted[buildingType][self.building2AddIdx].m_screenLocation
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(coord)]), False
            
        elif moveNum == 1:
            sc2Action = TerranUnit.BUILDING_SPEC[additionType].sc2Action

            if sc2Action in obs.observation['available_actions']:
                buildingCmdLocations = self.GetBuildingCmdAdditionLocations()
                coord = GetLocationForBuilding(obs, buildingType, self.notAllowedDirections2CC, buildingCmdLocations, additionType)
                if coord[0] >= 0:
                    buildCmd = BuildingCmdAddition(coord)
                    if additionType == Terran.BarracksReactor:
                        buildCmd.m_qHeadSize = 2

                    self.sharedData.buildCommands[additionType].append(buildCmd)
                    del self.sharedData.buildingCompleted[buildingType][self.building2AddIdx]
                    return actions.FunctionCall(sc2Action, [SC2_Params.QUEUED, SwapPnt(coord)]) , True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True 

    def SelectBuilding2Add(self, buildingType):
        if len(self.sharedData.buildingCompleted[buildingType]) > 0:
            return 0
        else:
            return -1

    def CreateState(self, obs):
        self.UpdateBuildingInProgress(obs)

        for key, value in BUILD_STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.NumBuildings(key)
            if value[1] >= 0:
                self.current_state[value[0]] += len(self.sharedData.buildCommands[key])
                self.current_state[value[1]] = len(self.sharedData.buildCommands[key])

        self.current_state[BUILD_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[BUILD_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        supplyUsed = obs.observation['player'][SC2_Params.SUPPLY_USED]
        supplyCap = obs.observation['player'][SC2_Params.SUPPLY_CAP]
        self.current_state[BUILD_STATE.SUPPLY_LEFT_IDX] = supplyCap - supplyUsed

        self.current_state[BUILD_STATE.TIME_LINE_IDX] = self.sharedData.numStep   

        self.ScaleState()

        if self.isActionCommitted:
            self.lastActionCommittedStep = self.sharedData.numAgentStep
            self.lastActionCommittedState = self.previous_scaled_state
            self.lastActionCommittedNextState = self.current_scaled_state
   
    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[BUILD_STATE.MINERALS_IDX] = min(BUILD_STATE.MINERALS_MAX, self.current_scaled_state[BUILD_STATE.MINERALS_IDX])
        self.current_scaled_state[BUILD_STATE.GAS_IDX] = min(BUILD_STATE.GAS_MAX, self.current_scaled_state[BUILD_STATE.GAS_IDX])

        self.current_scaled_state[BUILD_STATE.SUPPLY_LEFT_IDX] = min(BUILD_STATE.SUPPLY_LEFT_MAX, self.current_scaled_state[BUILD_STATE.SUPPLY_LEFT_IDX])

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions(self.current_scaled_state)
            else: 
                validActions = list(range(ACTIONS.NUM_ACTIONS))
 
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)
        else:
            action = ACTIONS.ID_DO_NOTHING
        
        self.current_action = action
        return action

    def ValidActions(self, state):
        valid = [ACTIONS.ID_DO_NOTHING]
        for key, requirement in self.actionsRequirement.items():
            if self.ValidSingleAction(state, requirement):
                valid.append(key)
        
        # special condition for oil refinery:
        if state[BUILD_STATE.REFINERY_IDX] + state[BUILD_STATE.IN_PROGRESS_REFINERY_IDX] >= self.maxNumOilRefinery and ACTIONS.ID_BUILD_REFINERY in valid:
            valid.remove(ACTIONS.ID_BUILD_REFINERY)

        # for addition building need building without addition
        for addition, building in ADDITION_2_BUILDING.items():
            action = ACTIONS.BUILDING_2_ACTION_TRANSITION[addition]
            if action in valid:
                numBuilding = state[BUILD_STATE.BUILDING_2_STATE_TRANSITION[building][0]]
                numAddition = state[BUILD_STATE.BUILDING_2_STATE_TRANSITION[addition][0]]
                numAddition += state[BUILD_STATE.BUILDING_2_STATE_TRANSITION[addition][1]]
                if numBuilding == numAddition:
                    valid.remove(action)

        for building, action in ACTIONS.BUILDING_2_ACTION_TRANSITION.items():
            if action in valid:
                if self.NumBeforeProgress(building) > 0:
                    valid.remove(action)

        return valid

    def ValidSingleAction(self, state, requirement):
        hasMinerals = state[BUILD_STATE.MINERALS_IDX] >= requirement.mineralsPrice
        hasGas = state[BUILD_STATE.GAS_IDX] >= requirement.gasPrice
        otherReq = state[requirement.buildingDependency] > 0
        return hasMinerals & hasGas & otherReq

    def NumBeforeProgress(self, building):
        num = 0
        for b in self.sharedData.buildCommands[building]:
            num += not b.m_inProgress
        
        return num


    def Action2Str(self, a, onlyAgent=False):
        return ACTIONS.ACTION2STR[a]

    def StateIdx2Str(self, idx):
        return BUILD_STATE.IDX2STR[idx]

    def PrintState(self):
        print(end = "[")
        for i in range(BUILD_STATE.SIZE):
            print(BUILD_STATE.IDX2STR[i], self.current_scaled_state[i], end = ', ')
        print("]")

    def UpdateBuildingInProgress(self, obs):
        for key in self.sharedData.buildCommands.keys():
            self.UpdateSingleBuilding(obs, key)

    def UpdateSingleBuilding(self, obs, buildingType):
        buildCmdVec = self.sharedData.buildCommands[buildingType]
        
        if buildingType in ADDITION_TYPES:
            additionCheck = True
            additionType = buildingType
            buildingType = ADDITION_2_BUILDING[buildingType]
        else:
            additionCheck = False

        for buildingCmd in buildCmdVec:
            if not buildingCmd.m_inProgress:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                xLoc = buildingCmd.m_screenLocation[SC2_Params.X_IDX]
                yLoc = buildingCmd.m_screenLocation[SC2_Params.Y_IDX]

                if unitType[yLoc][xLoc] == buildingType:            
                    buildingCmd.m_inProgress = True
                    buildingCmd.m_stepsCounter = 0 

                    # find building center location
                    buildingCmd.m_screenLocation = self.FindBuildingLocation(yLoc, xLoc, buildingType, unitType)
                    if additionCheck:
                        buildingCmd.m_additionCoord = self.FindAdditionLocation(yLoc, xLoc, buildingType, additionType, unitType)


                else:
                    buildingCmd.m_stepsCounter += 1
                    if buildingCmd.m_stepsCounter >= BUILDING_NOT_BUILT_THRESHOLD_COUNTER:
                        buildCmdVec.remove(buildingCmd)           
            else:
                buildingCmd.m_stepsCounter += 1             
    
    def FindBuildingLocation(self, y, x, buildingType, unitType):
        buildingMat = unitType == buildingType
        b_y, b_x = IsolateArea([y, x], buildingMat)
        return FindMiddle(b_y, b_x)

    def FindAdditionLocation(self, y, x, buildingType, additionType, unitType):
        
        while unitType[y][x] == buildingType and x + 1 < SC2_Params.SCREEN_SIZE:
            x += 1
        
        # if right edge is not addition ride n building up and down to search for addition
        if unitType[y][x] != additionType:
            yDown = y
            xDown = x - 1
            found = False
            while unitType[yDown][xDown] == buildingType:
                yDown += 1
                xDown = xDown if unitType[yDown][xDown] == buildingType else xDown + 1 if unitType[yDown][xDown + 1] == buildingType else xDown - 1
                if unitType[yDown][xDown + 1] == additionType:
                    y = yDown
                    x = xDown
                    found = True
                    break

            if not found:
                yUp = y
                xUp = x - 1
                while unitType[yUp][xUp] == buildingType:
                    yUp += 1
                    xUp = xUp if unitType[yUp][xUp] == buildingType else xUp + 1 if unitType[yUp][xUp + 1] == buildingType else xUp - 1
                    if unitType[yUp][xUp + 1] == additionType:
                        y = yUp
                        x = xUp
                        break
                    

        buildingMat = unitType == additionType
        b_y, b_x = IsolateArea([y, x], buildingMat)
        return FindMiddle(b_y, b_x)

    def NumBuildings(self, buildingType):
        num = len(self.sharedData.buildingCompleted[buildingType])
        if buildingType in BUILD_STATE.BUILDING_2_ADDITION:
            num += len(self.sharedData.buildingCompleted[BUILD_STATE.BUILDING_2_ADDITION[buildingType]])
            num += len(self.sharedData.buildCommands[BUILD_STATE.BUILDING_2_ADDITION[buildingType]])

        return num

    def RemoveDestroyedBuildings(self, obs, buildingType):
        pass

    def GetBuildingCmdAdditionLocations(self):
        allCoords = []

        for building, addition in BUILD_STATE.BUILDING_2_ADDITION.items():
            for b in self.sharedData.buildCommands[addition]:
                if not b.m_inProgress:
                    allCoords += self.InsertExpectedOccupiedLocations(b.m_screenLocation, building, addition)
            
        return allCoords
    
    def InsertExpectedOccupiedLocations(self, coord, buildingType, additionType):
        sizeBuilding = TerranUnit.BUILDING_SPEC[buildingType].screenPixels1Axis
        sizeAddition = TerranUnit.BUILDING_SPEC[additionType].screenPixels1Axis 

        coordNorthWest = [int(coord[0] - sizeBuilding / 2), int(coord[1] - sizeBuilding / 2)]
        coordSouthWest = [int(coord[0] + sizeBuilding / 2), int(coord[1] - sizeBuilding / 2)]
        coordSouthEast = [int(coord[0] + sizeBuilding / 2), int(coord[1] + sizeBuilding / 2)]
        coordAddition = [coord[0], int(coord[1] + sizeBuilding / 2 + sizeAddition)]
        
        return [coordNorthWest, coordSouthWest, coordSouthEast, coordAddition]

if __name__ == "__main__":
    from absl import app
    from absl import flags
    flags.DEFINE_string("directoryNames", "", "directory names to take results")
    flags.DEFINE_string("grouping", "100", "grouping size of results.")
    flags.FLAGS(sys.argv)

    directoryNames = (flags.FLAGS.directoryNames).split(",")
    grouping = int(flags.FLAGS.grouping)

    if "results" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping)