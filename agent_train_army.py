# train army sub agent
import random
import math
import os.path
import sys
from multiprocessing import Lock

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
from utils_decisionMaker import BaseDecisionMaker

from utils_results import ResultFile

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

from utils import EmptySharedData
from utils import SwapPnt
from utils import FindMiddle
from utils import GetScreenCorners
from utils import IsolateArea
from utils import Scale2MiniMap
from utils import GetUnitId
from utils import SelectBuildingValidPoint

from agent_build_base import ActionRequirement

AGENT_DIR = "TrainArmy/"
AGENT_NAME = "trainer"

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


ID_ACTION_DO_NOTHING = 0
ID_ACTION_TRAIN_MARINE = 1
ID_ACTION_TRAIN_REAPER = 2
ID_ACTION_TRAIN_HELLION = 3
ID_ACTION_TRAIN_SIEGETANK = 4

NUM_ACTIONS = 5

ACTION2STR = ["DoNothing" , "TrainMarine", "TrainReaper", "TrainHellion", "TrainSiegeTank"]

ACTION_2_UNIT = {}
ACTION_2_UNIT[ID_ACTION_TRAIN_MARINE] = GetUnitId("marine")
ACTION_2_UNIT[ID_ACTION_TRAIN_REAPER] = GetUnitId("reaper")
ACTION_2_UNIT[ID_ACTION_TRAIN_HELLION] = GetUnitId("hellion")
ACTION_2_UNIT[ID_ACTION_TRAIN_SIEGETANK] = GetUnitId("siege tank")

# state details
STATE_NON_VALID_NUM = -1

STATE_MINERALS_MAX = 500
STATE_GAS_MAX = 300
STATE_MINERALS_BUCKETING = 50
STATE_GAS_BUCKETING = 50

STATE_MINERALS_IDX = 0
STATE_GAS_IDX = 1
STATE_SUPPLY_DEPOT_IDX = 2
STATE_BARRACKS_IDX = 3
STATE_FACTORY_IDX = 4
STATE_REACTORS_IDX = 5
STATE_TECHLAB_IDX = 6
STATE_ARMY_POWER = 7
STATE_QUEUE_BARRACKS = 8
STATE_QUEUE_FACTORY = 9
STATE_QUEUE_FACTORY_WITH_TECHLAB = 10
STATE_SIZE = 11

STATE_IDX2STR = ["min", "gas", "sd", "ba", "fa", "re", "te", "power", "ba_q", "fa_q", "te_q"]

BUILDING_2_STATE_TRANSITION = {}
BUILDING_2_STATE_TRANSITION[Terran.SupplyDepot] = STATE_SUPPLY_DEPOT_IDX
BUILDING_2_STATE_TRANSITION[Terran.Barracks] = STATE_BARRACKS_IDX
BUILDING_2_STATE_TRANSITION[Terran.Factory] = STATE_FACTORY_IDX
BUILDING_2_STATE_TRANSITION[Terran.Reactor] = STATE_REACTORS_IDX
BUILDING_2_STATE_TRANSITION[Terran.TechLab] = STATE_TECHLAB_IDX

BUILDING_2_STATE_QUEUE_TRANSITION = {}

BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Barracks] = STATE_QUEUE_BARRACKS
BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Factory] = STATE_QUEUE_FACTORY
BUILDING_2_STATE_QUEUE_TRANSITION[Terran.TechLab] = STATE_QUEUE_FACTORY_WITH_TECHLAB

class SharedDataTrain(EmptySharedData):
    def __init__(self):
        super(SharedDataTrain, self).__init__()
        self.trainingQueue = {}
        for key in BUILDING_2_STATE_QUEUE_TRANSITION.keys():
            self.trainingQueue[key] = []

        self.armySize = {}
        self.unitTrainValue = {}
        for unit in ACTION_2_UNIT.values():
            self.armySize[unit] = 0
            self.unitTrainValue[unit] = 0.0

        self.prevTrainActionReward = 0.0



class BuildingDetailsForProduction:
    def __init__(self, screenCoord):
        self.screenCoord = screenCoord
        self.qSize = 0
        self.hasAddition = False


class TrainCmd:
    def __init__(self, unitId):
        self.unitId = unitId
        self.stepsCounter = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DIRECTORY] = "trainArmy_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "train_dqn"
RUN_TYPES[DQN][DIRECTORY] = "trainArmy_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "trainArmy_naive"
RUN_TYPES[NAIVE][RESULTS] = "trainArmy_result"

class NaiveDecisionMakerTrain(BaseDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Learn = 20):
        super(NaiveDecisionMakerTrain, self).__init__()
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

    def ActionValuesVec(self, state, target = True):    
        vals = np.zeros(NUM_ACTIONS,dtype = float)
        
        factoryQ = state[STATE_QUEUE_FACTORY] + state[STATE_QUEUE_FACTORY_WITH_TECHLAB]
        if state[STATE_QUEUE_BARRACKS] > factoryQ:
            vals[ID_ACTION_TRAIN_MARINE] = 0.5
            vals[ID_ACTION_TRAIN_REAPER] = 0.4
            vals[ID_ACTION_TRAIN_SIEGETANK] = 0.3
            vals[ID_ACTION_TRAIN_HELLION] = 0.25            
        else:
            vals[ID_ACTION_TRAIN_MARINE] = 0.25
            vals[ID_ACTION_TRAIN_REAPER] = 0.1
            r = np.random.uniform()
            if r > 0.25:
                vals[ID_ACTION_TRAIN_SIEGETANK] = 0.5
                vals[ID_ACTION_TRAIN_HELLION] = 0.4  
            else:
                vals[ID_ACTION_TRAIN_SIEGETANK] = 0.4 
                vals[ID_ACTION_TRAIN_HELLION] = 0.5  


        vals[self.choose_action(state)] = 1.0

        return vals


class TrainArmySubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):     
        super(TrainArmySubAgent, self).__init__(STATE_SIZE)     

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList
        
        self.illigalmoveSolveInModel = True

        # tables:
        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.sharedData = sharedData

        # model params
        self.actionsRequirement = {}
        self.actionsRequirement[ID_ACTION_TRAIN_MARINE] = ActionRequirement(50, 0, Terran.Barracks)
        self.actionsRequirement[ID_ACTION_TRAIN_REAPER] = ActionRequirement(50, 50, Terran.Barracks)
        self.actionsRequirement[ID_ACTION_TRAIN_HELLION] = ActionRequirement(100, 0, Terran.Factory)
        self.actionsRequirement[ID_ACTION_TRAIN_SIEGETANK] = ActionRequirement(150, 125, Terran.TechLab)

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        runType = RUN_TYPES[dmTypes[AGENT_NAME]]
        # create agent dir
        directory = dmTypes["directory"] + "/" + AGENT_DIR
        if not os.path.isdir("./" + directory):
            os.makedirs("./" + directory)

        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerTrain(resultFName=runType[RESULTS], directory=directory + runType[DIRECTORY])
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

    def FirstStep(self, obs):
        super(TrainArmySubAgent, self).FirstStep(obs)   

        # states and action:
        self.current_action = None
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.numSteps = 0

        self.countBuildingsQueue = {}
        for key in BUILDING_2_STATE_QUEUE_TRANSITION.keys():
            self.countBuildingsQueue = []

    def IsDoNothingAction(self, a):
        return a == ID_ACTION_DO_NOTHING

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

    def Action2SC2Action(self, obs, a, moveNum):
        if moveNum == 0:
            finishedAction = False
            buildingType = self.BuildingType(a)
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            target = SelectBuildingValidPoint(unitType, buildingType)
            if target[0] >= 0:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), finishedAction
        if moveNum == 2:
            finishedAction = True
            unit2Train = ACTION_2_UNIT[a]
            sc2Action = TerranUnit.ARMY_SPEC[unit2Train].sc2Action
            if sc2Action in obs.observation['available_actions']:
                self.sharedData.prevTrainActionReward = self.sharedData.unitTrainValue[unit2Train]
                buildingReq4Train = self.actionsRequirement[a].buildingDependency
                self.sharedData.trainingQueue[buildingReq4Train].append(TrainCmd(unit2Train))
                return actions.FunctionCall(sc2Action, [SC2_Params.QUEUED]), finishedAction

        return DO_NOTHING_SC2_ACTION, True

    def UpdateNewBuildings(self, obs):
        self.UpdateSingleBuilding(obs, Terran.Barracks)
        self.UpdateSingleBuilding(obs, Terran.Factory)

    def UpdateSingleBuilding(self, obs, buildingType):
        stateIdx = BUILDING_2_STATE_TRANSITION[buildingType]
        currNum = self.current_scaled_state[stateIdx]
        prevNum = self.previous_scaled_state[stateIdx]
        if currNum == prevNum:
            return   
        elif currNum > prevNum:
            print("\ntrain agent found new buildings")
            exit()
            for i in range(prevNum, currNum):
                self.AddNewBuilding(obs, Terran.Barracks)
        else:
            self.RemoveDestroyedBuildings(obs, Terran.Barracks)

    def AddNewBuilding(self, obs, buildingType):
        unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        buildingMat = unitType == buildingType
        for building in self.countBuildingsQueue[buildingType]:
            b_y, b_x = IsolateArea(building.screenCoord, buildingMat)
            buildingMat[b_y][b_x] = False

        new_y, new_x = buildingMat.nonzero()
        if len(new_y) > 0:
            newSingle_y, newSingle_x = IsolateArea((new_y[0], new_x[0]), buildingMat)
            midPnt = FindMiddle(newSingle_y, newSingle_x)
            self.countBuildingsQueue[buildingType].append(BuildingDetailsForProduction(midPnt))
        else:
            print("didn't found new building")

    def RemoveDestroyedBuildings(self, obs, buildingType):
        pass

    def Learn(self, reward, terminal):            
        if self.trainAgent:
            if self.isActionCommitted:
                self.decisionMaker.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(NUM_ACTIONS):
                    self.decisionMaker.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.decisionMaker.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def CreateState(self, obs):
        self.UpdateNewBuildings(obs)
        
        self.current_state[STATE_MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[STATE_GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]
        for key, value in BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value] = self.sharedData.buildingCount[key]

        for key, value in BUILDING_2_STATE_QUEUE_TRANSITION.items():
            self.current_state[value] = len(self.sharedData.trainingQueue[key])
        
        power = 0.0
        for unit, num in self.sharedData.armySize.items():
            power += num * self.sharedData.unitTrainValue[unit]
        
        self.current_state[STATE_ARMY_POWER] = round(power)

        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]

        self.current_scaled_state[STATE_MINERALS_IDX] = int(self.current_scaled_state[STATE_MINERALS_IDX] / STATE_MINERALS_BUCKETING) * STATE_MINERALS_BUCKETING
        self.current_scaled_state[STATE_MINERALS_IDX] = min(STATE_MINERALS_MAX, self.current_scaled_state[STATE_MINERALS_IDX])
        self.current_scaled_state[STATE_GAS_IDX] = int(self.current_scaled_state[STATE_GAS_IDX] / STATE_GAS_BUCKETING) * STATE_GAS_BUCKETING
        self.current_scaled_state[STATE_GAS_IDX] = min(STATE_GAS_MAX, self.current_scaled_state[STATE_GAS_IDX])

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

        return action

    def ValidActions(self):
        valid = [ID_ACTION_DO_NOTHING]
        for key, requirement in self.actionsRequirement.items():
            if self.ValidSingleAction(requirement):
                valid.append(key)
        return valid

    def ValidSingleAction(self, requirement):
        hasMinerals = self.current_scaled_state[STATE_MINERALS_IDX] >= requirement.mineralsPrice
        hasGas = self.current_scaled_state[STATE_GAS_IDX] >= requirement.gasPrice
        idx = BUILDING_2_STATE_TRANSITION[requirement.buildingDependency]
        otherReq = self.current_scaled_state[idx] > 0
        return hasMinerals & hasGas & otherReq

    def BuildingType(self, action):
        if action > ID_ACTION_DO_NOTHING:
            if action > ID_ACTION_TRAIN_REAPER:
                return Terran.Factory
            else:
                return Terran.Barracks
        else:
            return None

    def Action2Str(self,a):
        return ACTION2STR[a]

    def PrintState(self):
        for i in range(STATE_SIZE):
            print(STATE_IDX2STR[i], self.current_scaled_state[i], end = ', ')

        print("")
