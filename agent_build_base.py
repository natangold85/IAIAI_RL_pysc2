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
from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay
from utils_decisionMaker import BaseDecisionMaker

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
from utils import IsolateArea
from utils import SelectBuildingValidPoint
from utils import SelectUnitValidPoints


AGENT_DIR = "BuildBase/"
if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)
AGENT_NAME = "builder"

# possible types of decision maker

QTABLE = 'q'
DQN = 'dqn'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding' 
NAIVE = "naive"
USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, DQN_EMBEDDING_LOCATIONS, NAIVE])

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
    BUILDING_2_ACTION_TRANSITION[Terran.SupplyDepot] = ID_BUILD_SUPPLY_DEPOT
    BUILDING_2_ACTION_TRANSITION[Terran.Refinery] = ID_BUILD_REFINERY
    BUILDING_2_ACTION_TRANSITION[Terran.Barracks] = ID_BUILD_BARRACKS
    BUILDING_2_ACTION_TRANSITION[Terran.Factory] = ID_BUILD_FACTORY

    ACTION_2_BUILDING_TRANSITION = {}
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_SUPPLY_DEPOT] = Terran.SupplyDepot
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_REFINERY] = Terran.Refinery
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_BARRACKS] = Terran.Barracks
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_FACTORY] = Terran.Factory
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_BARRACKS_REACTOR] = [Terran.Barracks, Terran.Reactor]
    ACTION_2_BUILDING_TRANSITION[ID_BUILD_FACTORY_TECHLAB] = [Terran.Factory, Terran.TechLab]

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
    BUILDING_2_STATE_TRANSITION[Terran.CommandCenter] = [COMMAND_CENTER_IDX, -1]
    BUILDING_2_STATE_TRANSITION[Terran.SupplyDepot] = [SUPPLY_DEPOT_IDX, IN_PROGRESS_SUPPLY_DEPOT_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Refinery] = [REFINERY_IDX, IN_PROGRESS_REFINERY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Barracks] = [BARRACKS_IDX, IN_PROGRESS_BARRACKS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Factory] = [FACTORY_IDX, IN_PROGRESS_FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Reactor] = [REACTORS_IDX, IN_PROGRESS_RECTORS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.TechLab] = [TECHLAB_IDX, IN_PROGRESS_TECHLAB_IDX]

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

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "buildBase_naive"
RUN_TYPES[NAIVE][RESULTS] = "buildBase_result"

class SharedDataBuild(EmptySharedData):
    def __init__(self):
        super(SharedDataBuild, self).__init__()
        self.buildCommands = {}
        self.buildingCount = {}
        for key in STATE.BUILDING_2_STATE_TRANSITION.keys():
            self.buildingCount[key] = 0
            self.buildCommands[key] = []

        self.commandCenterLoc = []

class BuildingCmd:
    def __init__(self, screenLocation, inProgress = False):
        self.m_screenLocation = screenLocation
        self.m_inProgress = inProgress
        self.m_stepsCounter = 0

class NaiveDecisionMakerBuilder(BaseDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Learn = 20):
        super(NaiveDecisionMakerBuilder, self).__init__()
        self.resultFName = resultFName
        self.trialNum = 0
        self.numTrials2Learn = numTrials2Learn

        if resultFName != None:
            self.lock = Lock()

            if directory != None:
                fullDirectoryName = "./" + directory +"/"
            else:
                fullDirectoryName = "./"

            if "new" in sys.argv:
                loadFiles = False
            else:
                loadFiles = True

            self.resultFile = ResultFile(fullDirectoryName + resultFName, numTrials2Learn, loadFiles)


    def choose_action(self, state):
        action = 0

        numSDAll = state[STATE.SUPPLY_DEPOT_IDX] + state[STATE.IN_PROGRESS_SUPPLY_DEPOT_IDX]
        numRefAll = state[STATE.REFINERY_IDX] + state[STATE.IN_PROGRESS_REFINERY_IDX]
        numBaAll = state[STATE.BARRACKS_IDX] + state[STATE.IN_PROGRESS_BARRACKS_IDX]
        numFaAll = state[STATE.FACTORY_IDX] + state[STATE.IN_PROGRESS_FACTORY_IDX]
        numReactorsAll = state[STATE.REACTORS_IDX] + state[STATE.IN_PROGRESS_RECTORS_IDX]
        numTechAll = state[STATE.TECHLAB_IDX] + state[STATE.IN_PROGRESS_TECHLAB_IDX]
        
        if numSDAll < 2:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT
        elif numRefAll < 2:
            action = ACTIONS.ID_BUILD_REFINERY
        elif numBaAll < 2:
            action = ACTIONS.ID_BUILD_BARRACKS
        elif numFaAll < 1:
            action = ACTIONS.ID_BUILD_FACTORY
        elif numReactorsAll < 2:
            action = ACTIONS.ID_BUILD_BARRACKS_REACTOR
        elif numSDAll < 5:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT
        elif numFaAll < 2:
            action = ACTIONS.ID_BUILD_FACTORY
        elif numTechAll < 2:
            action = ACTIONS.ID_BUILD_FACTORY_TECHLAB
        elif numBaAll < 4:
            action = ACTIONS.ID_BUILD_BARRACKS
        elif numSDAll < 7:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT
        elif numFaAll < 3:
            action = ACTIONS.ID_BUILD_FACTORY
        elif numTechAll < 3:
            action = ACTIONS.ID_BUILD_FACTORY_TECHLAB
        else:
            action = ACTIONS.ID_BUILD_SUPPLY_DEPOT

        return action

    def ActionValuesVec(self, state, target = True):    
        vals = np.zeros(ACTIONS.NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state)] = 1.0
        vals[ACTIONS.ID_DO_NOTHING] = 0.1
        return vals

    def end_run(self, r, score, steps):
        saveFile = False
        self.trialNum += 1
        
        if self.resultFName != None:
            self.lock.acquire()
            if self.trialNum % self.numTrials2Learn == 0:
                saveFile = True

            self.resultFile.end_run(r, score, steps, saveFile)
            self.lock.release()
       
        return saveFile 

class BuildBaseSubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):        
        super(BuildBaseSubAgent, self).__init__(STATE.SIZE)
        self.discountFactor = 0.95
        
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent =AGENT_NAME in trainList
        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        self.sharedData = sharedData

        # states and action:
        self.terminalState = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        self.actionsRequirement = {}
        self.actionsRequirement[ACTIONS.ID_BUILD_SUPPLY_DEPOT] = ActionRequirement(100)
        self.actionsRequirement[ACTIONS.ID_BUILD_REFINERY] = ActionRequirement(75)
        self.actionsRequirement[ACTIONS.ID_BUILD_BARRACKS] = ActionRequirement(150,0,STATE.SUPPLY_DEPOT_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_FACTORY] = ActionRequirement(150,100,STATE.BARRACKS_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_BARRACKS_REACTOR] = ActionRequirement(50,50,STATE.BARRACKS_IDX)
        self.actionsRequirement[ACTIONS.ID_BUILD_FACTORY_TECHLAB] = ActionRequirement(50,25,STATE.FACTORY_IDX)

        self.maxNumOilRefinery = 2

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        runType = RUN_TYPES[dmTypes[AGENT_NAME]]
        # create agent dir
        directory = dmTypes["directory"] + "/" + AGENT_DIR
        if not os.path.isdir("./" + directory):
            os.makedirs("./" + directory)

        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerBuilder(resultFName=runType[RESULTS], directory=directory + runType[DIRECTORY])
        else:        
            decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                                resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory + runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

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

    def FirstStep(self, obs):
        super(BuildBaseSubAgent, self).FirstStep()

        self.current_action = None

        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        self.stepActionCommmitted = 0
        self.lastActionCommittedState = None
        self.lastActionCommittedNextState = None

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

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

    def IsDoNothingAction(self, a):
        return a == ACTIONS.ID_DO_NOTHING

    def BuildAction(self, obs, moveNum):
        if moveNum == 0:
            finishedAction = False
            # select scv
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            unit_y, unit_x = (unitType == Terran.SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, target]), finishedAction
        
        elif moveNum == 1:
            finishedAction = False
            buildingType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action]
            sc2Action = TerranUnit.BUILDING_SPEC[buildingType].sc2Action

            # preform build action. if action not available go back to harvest
            if sc2Action in obs.observation['available_actions']:
                coord = GetLocationForBuilding(obs, buildingType)
                if coord[SC2_Params.Y_IDX] >= 0:
                    self.sharedData.buildCommands[buildingType].append(BuildingCmd(coord))
                    return actions.FunctionCall(sc2Action, [SC2_Params.NOT_QUEUED, SwapPnt(coord)]), finishedAction
            moveNum += 1

        if moveNum == 2:
            finishedAction = True
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = self.GatherHarvest(unitType)
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), finishedAction
     
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def BuildAdditionAction(self, obs, moveNum):
        buildingType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action][0]
        additionType = ACTIONS.ACTION_2_BUILDING_TRANSITION[self.current_action][1]

        if moveNum == 0:
            # select building without addition
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            target = SelectBuildingValidPoint(unitType, buildingType)
            if target[0] >= 0:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), False
            
        elif moveNum == 1:
            action = TerranUnit.BUILDING_SPEC[additionType].sc2Action
            if action in obs.observation['available_actions']:
                coord = GetLocationForBuilding(obs, buildingType, additionType)
                if coord[0] >= 0:
                    
                    numBuildingsCompleted = self.current_state[STATE.BUILDING_2_STATE_TRANSITION[buildingType][0]]
                    numAdditionAll = sum(self.current_state[STATE.BUILDING_2_STATE_TRANSITION[additionType]])
                    if numBuildingsCompleted > numAdditionAll:
                        self.sharedData.buildCommands[additionType].append(BuildingCmd(coord))
                        return actions.FunctionCall(action, [SC2_Params.QUEUED, SwapPnt(coord)]) , True
                #     else:
                #         print("failed in num addition")
                # else:
                #     print("failed in num location")

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True    

    def CreateState(self, obs):
        for key, value in STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.sharedData.buildingCount[key]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.sharedData.buildCommands[key])
    
        self.current_state[STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        self.ScaleState()

        # if self.isActionCommitted:
        #     self.stepActionCommmitted = self.numSteps
        #     self.lastActionCommittedState = self.previous_scaled_state
        #     self.lastActionCommittedNextState = self.current_scaled_state
   
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
                    exploreProb = self.decisionMaker.TargetExploreProb()   

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
            action = ACTIONS.ID_DO_NOTHING
        
        self.current_action = action
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
    
    def GatherHarvest(self, unitType):
        if random.randint(0, 2) < 1:
            resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD
        else:
            resourceList = [Terran.Refinery]

        allResMat = np.in1d(unitType, resourceList).reshape(unitType.shape)
        unit_y, unit_x = SelectUnitValidPoints(allResMat)
        if len(unit_y) > 0:
            i = random.randint(0, len(unit_y) - 1)
            return [unit_y[i], unit_x[i]]
        
        return [-1,-1]

    def Action2Str(self, a):
        return ACTIONS.ACTION2STR[a]
