# train army sub agent
import random
import math
import os.path
import sys
import time
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

ACTION_2_BUILDING = {}
ACTION_2_BUILDING[ID_ACTION_DO_NOTHING] = None
ACTION_2_BUILDING[ID_ACTION_TRAIN_MARINE] = Terran.Barracks
ACTION_2_BUILDING[ID_ACTION_TRAIN_REAPER] = Terran.Barracks
ACTION_2_BUILDING[ID_ACTION_TRAIN_HELLION] = Terran.Factory
ACTION_2_BUILDING[ID_ACTION_TRAIN_SIEGETANK] = Terran.Factory

ACTION_2_ADDITION = {}
ACTION_2_ADDITION[ID_ACTION_TRAIN_SIEGETANK] = Terran.FactoryTechLab

class TRAIN_STATE:
    # state details
    NON_EXIST_Q_VAL = 10

    MINERALS_MAX = 500
    GAS_MAX = 300
    MINERALS_BUCKETING = 50
    GAS_BUCKETING = 50

    MINERALS_IDX = 0
    GAS_IDX = 1
    SUPPLY_DEPOT_IDX = 2
    BARRACKS_IDX = 3
    FACTORY_IDX = 4
    REACTORS_IDX = 5
    TECHLAB_IDX = 6

    QUEUE_BARRACKS = 8
    QUEUE_FACTORY = 9
    QUEUE_FACTORY_WITH_TECHLAB = 10
 
    SIZE = 11

    IDX2STR = ["min", "gas", "sd", "ba", "fa", "re", "te", "power", "ba_q", "fa_q", "te_q"]

    BUILDING_2_STATE_TRANSITION = {}
    BUILDING_2_STATE_TRANSITION[Terran.SupplyDepot] = SUPPLY_DEPOT_IDX
    BUILDING_2_STATE_TRANSITION[Terran.Barracks] = BARRACKS_IDX
    BUILDING_2_STATE_TRANSITION[Terran.Factory] = FACTORY_IDX
    BUILDING_2_STATE_TRANSITION[Terran.BarracksReactor] = REACTORS_IDX
    BUILDING_2_STATE_TRANSITION[Terran.FactoryTechLab] = TECHLAB_IDX

    BUILDING_2_STATE_QUEUE_TRANSITION = {}

    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Barracks] = QUEUE_BARRACKS
    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Factory] = QUEUE_FACTORY
    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.FactoryTechLab] = QUEUE_FACTORY_WITH_TECHLAB

    BUILDING_2_ADDITION = {}
    BUILDING_2_ADDITION[Terran.Barracks] = Terran.BarracksReactor
    BUILDING_2_ADDITION[Terran.Factory] = Terran.FactoryTechLab

    DOUBLE_QUEUE_ADDITION = [Terran.BarracksReactor, Terran.FactoryReactor, Terran.Reactor]

class SharedDataTrain(EmptySharedData):
    def __init__(self):
        super(SharedDataTrain, self).__init__()

        self.armySize = {}
        self.unitTrainValue = {}
        for unit in ACTION_2_UNIT.values():
            self.armySize[unit] = 0
            self.unitTrainValue[unit] = 0.0

        self.prevTrainActionReward = 0.0

        self.qMinSizes = {}

class TrainCmd:
    def __init__(self, unitId):
        self.unit = unitId
        self.step = -1000

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(TRAIN_STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DIRECTORY] = "trainArmy_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(TRAIN_STATE.SIZE, NUM_ACTIONS)
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
                if not os.path.isdir(fullDirectoryName):
                    os.makedirs(fullDirectoryName)
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
        
        if state[TRAIN_STATE.QUEUE_BARRACKS] < state[TRAIN_STATE.QUEUE_FACTORY] and state[TRAIN_STATE.QUEUE_BARRACKS] < state[TRAIN_STATE.QUEUE_FACTORY_WITH_TECHLAB]:
            vals[ID_ACTION_TRAIN_MARINE] = 0.5
            vals[ID_ACTION_TRAIN_REAPER] = 0.4
            vals[ID_ACTION_TRAIN_SIEGETANK] = 0.3
            vals[ID_ACTION_TRAIN_HELLION] = 0.25                    
        else:
            vals[ID_ACTION_TRAIN_MARINE] = 0.25
            vals[ID_ACTION_TRAIN_REAPER] = 0.1
            r = np.random.uniform()
            if r > 0.25 and state[TRAIN_STATE.QUEUE_FACTORY] > state[TRAIN_STATE.QUEUE_FACTORY_WITH_TECHLAB]:
                vals[ID_ACTION_TRAIN_SIEGETANK] = 0.5
                vals[ID_ACTION_TRAIN_HELLION] = 0.4  
            else:
                vals[ID_ACTION_TRAIN_SIEGETANK] = 0.4 
                vals[ID_ACTION_TRAIN_HELLION] = 0.5  


        vals[self.choose_action(state)] = 1.0

        return vals


class TrainArmySubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):     
        super(TrainArmySubAgent, self).__init__(TRAIN_STATE.SIZE)     

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
        self.actionsRequirement[ID_ACTION_TRAIN_SIEGETANK] = ActionRequirement(150, 125, Terran.FactoryTechLab)

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        runType = RUN_TYPES[dmTypes[AGENT_NAME]]
        # create agent dir
        directory = dmTypes["directory"] + "/" + AGENT_DIR

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
        self.current_state = np.zeros(TRAIN_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(TRAIN_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(TRAIN_STATE.SIZE, dtype=np.int32, order='C')

        self.numSteps = 0

        self.countBuildingsQueue = {}
        for key in TRAIN_STATE.BUILDING_2_STATE_QUEUE_TRANSITION.keys():
            self.countBuildingsQueue = []

    def IsDoNothingAction(self, a):
        return a == ID_ACTION_DO_NOTHING

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

    def Action2SC2Action(self, obs, a, moveNum):
        if moveNum == 0:
            finishedAction = False
            self.buildingSelected = self.SelectBuilding2Train(a)
            if self.buildingSelected != None:
                target = self.buildingSelected.m_screenLocation
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), finishedAction
        
        elif moveNum == 1:
            finishedAction = True
            unit2Train = ACTION_2_UNIT[a]
            sc2Action = TerranUnit.ARMY_SPEC[unit2Train].sc2Action
            if sc2Action in obs.observation['available_actions']:
                self.sharedData.prevTrainActionReward = self.sharedData.unitTrainValue[unit2Train]
                self.Add2Q(a)
                return actions.FunctionCall(sc2Action, [SC2_Params.QUEUED]), finishedAction

        return DO_NOTHING_SC2_ACTION, True

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

        self.current_state[TRAIN_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[TRAIN_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]
        for key, value in TRAIN_STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value] = self.NumBuildings(key)

        self.sharedData.qMinSizes = self.CalculateQueues()
        for key, value in TRAIN_STATE.BUILDING_2_STATE_QUEUE_TRANSITION.items():
            if key in self.sharedData.qMinSizes:
                self.current_state[value] = self.sharedData.qMinSizes[key]
            else:
                self.current_state[value] = TRAIN_STATE.NON_EXIST_Q_VAL
        
        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]

        self.current_scaled_state[TRAIN_STATE.MINERALS_IDX] = int(self.current_scaled_state[TRAIN_STATE.MINERALS_IDX] / TRAIN_STATE.MINERALS_BUCKETING) * TRAIN_STATE.MINERALS_BUCKETING
        self.current_scaled_state[TRAIN_STATE.MINERALS_IDX] = min(TRAIN_STATE.MINERALS_MAX, self.current_scaled_state[TRAIN_STATE.MINERALS_IDX])
        self.current_scaled_state[TRAIN_STATE.GAS_IDX] = int(self.current_scaled_state[TRAIN_STATE.GAS_IDX] / TRAIN_STATE.GAS_BUCKETING) * TRAIN_STATE.GAS_BUCKETING
        self.current_scaled_state[TRAIN_STATE.GAS_IDX] = min(TRAIN_STATE.GAS_MAX, self.current_scaled_state[TRAIN_STATE.GAS_IDX])

    def NumBuildings(self, buildingType):
        num = len(self.sharedData.buildingCompleted[buildingType])
        if buildingType in TRAIN_STATE.BUILDING_2_ADDITION:
            num += len(self.sharedData.buildingCompleted[TRAIN_STATE.BUILDING_2_ADDITION[buildingType]])
            num += len(self.sharedData.buildCommands[TRAIN_STATE.BUILDING_2_ADDITION[buildingType]])

        return num

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
        hasMinerals = self.current_scaled_state[TRAIN_STATE.MINERALS_IDX] >= requirement.mineralsPrice
        hasGas = self.current_scaled_state[TRAIN_STATE.GAS_IDX] >= requirement.gasPrice
        idx = TRAIN_STATE.BUILDING_2_STATE_TRANSITION[requirement.buildingDependency]
        otherReq = self.current_scaled_state[idx] > 0
        return hasMinerals & hasGas & otherReq

    def Action2Str(self,a):
        return ACTION2STR[a]

    def PrintState(self):
        for i in range(TRAIN_STATE.SIZE):
            print(TRAIN_STATE.IDX2STR[i], self.current_scaled_state[i], end = ', ')

        print("")

    def GetStateVal(self, idx):
        return self.current_state[idx]

    def SelectBuilding2Train(self, action):
        buildingType = ACTION_2_BUILDING[action]
        additionType = TRAIN_STATE.BUILDING_2_ADDITION[buildingType]
        
        needAddition = action in ACTION_2_ADDITION.keys()
        
        qHeadSize = 1 + 1 * (additionType in TRAIN_STATE.DOUBLE_QUEUE_ADDITION)
        

        buildingShortestQAdd, shortestQAdd = self.FindMinQueue(additionType)

        shortestQAdd = int(shortestQAdd / qHeadSize)
        if not needAddition:
            buildingShortestQ, shortestQ = self.FindMinQueue(buildingType)
            if shortestQ < shortestQAdd:
                buildingChosen = buildingShortestQ
            else:
                buildingChosen = buildingShortestQAdd
        else:
            buildingChosen = buildingShortestQAdd

        return buildingChosen
    
    def FindMinQueue(self, buildingType):
        shortestQ = 100
        buildingChosen = None

        for b in self.sharedData.buildingCompleted[buildingType]:
            q = len(b.qForProduction)                      
            if q < shortestQ:
                shortestQ = q
                buildingChosen = b

        return buildingChosen, shortestQ


    def Add2Q(self, action):
        unit = TrainCmd(ACTION_2_UNIT[action])
        self.buildingSelected.AddUnit2Q(unit)

    def CalculateQueues(self):
        qMinSizes = {}

        minQBarracks, validB = self.CalculateMinQ(Terran.Barracks)
        # for barracks building reactor q is 2 times faster
        minQBarracksReactor, validBR = self.CalculateMinQ(Terran.BarracksReactor)
        minQBarracksReactor = int(minQBarracksReactor / 2)

        minQFactory, validF = self.CalculateMinQ(Terran.Factory)
        minQFactoryTL, validFTL = self.CalculateMinQ(Terran.FactoryTechLab)

        # calc Q sizes
        if validB or validBR:
            qMinSizes[Terran.Barracks] = min(minQBarracks, minQBarracksReactor)

        if validF or validFTL:
            qMinSizes[Terran.Factory] = min(minQFactory, minQFactoryTL)   
            if validFTL:
                qMinSizes[Terran.FactoryTechLab] = minQFactoryTL


        return qMinSizes

    def CalculateMinQ(self, buildingType):
        minQ = 1000
        valid = False
        for b in self.sharedData.buildingCompleted[buildingType]:
            b.AddStepForUnits()
            q = len(b.qForProduction)
            if q < minQ:
                minQ = q
                valid = True
      
        return minQ, valid 