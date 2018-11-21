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

from utils import EmptySharedData
from utils import SupplyCap
from utils import SwapPnt
from utils import FindMiddle
from utils import GetScreenCorners
from utils import IsolateArea
from utils import Scale2MiniMap
from utils import GetUnitId
from utils import SelectBuildingValidPoint

from agent_build_base import ActionRequirement

AGENT_DIR = "TrainArmy/"
AGENT_NAME = "train_army"

# possible types of decision maker

QTABLE = 'q'
DQN = 'dqn'
DQN2L = 'dqn_2l' 
DQN2L_DFLT = 'dqn_2l_dflt'
A2C = "A2C"
NAIVE = "naive"
USER_PLAY = 'play'

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "history"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

# Model Params
NUM_TRIALS_2_LEARN = 20
NUM_TRIALS_4_CMP = 200


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
    
    SUPPLY_LEFT_MAX = 10
    SUPPLY_LEFT_BUCKETING = 2

    MINERALS_IDX = 0
    GAS_IDX = 1
    BARRACKS_IDX = 2
    FACTORY_IDX = 3
    REACTORS_IDX = 4
    TECHLAB_IDX = 5

    SUPPLY_LEFT = 6

    QUEUE_BARRACKS = 7
    QUEUE_FACTORY = 8
    QUEUE_FACTORY_WITH_TECHLAB = 9
    
    ARMY_POWER = 10

    TIME_LINE_IDX = 11
    
    SIZE = 12

    IDX2STR = {}
    IDX2STR[MINERALS_IDX] = "min"
    IDX2STR[GAS_IDX] = "gas"
    IDX2STR[BARRACKS_IDX] = "ba"
    IDX2STR[FACTORY_IDX] = "fa"
    IDX2STR[REACTORS_IDX] = "re"
    IDX2STR[TECHLAB_IDX] = "tl"
    IDX2STR[SUPPLY_LEFT] = "supp_left"
    IDX2STR[QUEUE_BARRACKS] = "ba_q"
    IDX2STR[QUEUE_FACTORY] = "fa_q"
    IDX2STR[QUEUE_FACTORY_WITH_TECHLAB] = "tl_q"

    IDX2STR[ARMY_POWER] = "power"
    IDX2STR[TIME_LINE_IDX] = "TimeLine"

    BUILDING_2_STATE_TRANSITION = {}
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
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(TRAIN_STATE.SIZE, NUM_ACTIONS, numTrials2Learn=NUM_TRIALS_2_LEARN)
RUN_TYPES[QTABLE][DIRECTORY] = "trainArmy_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(TRAIN_STATE.SIZE, NUM_ACTIONS, numTrials2Learn=NUM_TRIALS_2_LEARN)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "train_dqn"
RUN_TYPES[DQN][DIRECTORY] = "trainArmy_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

RUN_TYPES[DQN2L] = {}
RUN_TYPES[DQN2L][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN2L][PARAMS] = DQN_PARAMS(TRAIN_STATE.SIZE, NUM_ACTIONS, layersNum=2, numTrials2CmpResults=NUM_TRIALS_4_CMP, numTrials2Learn=NUM_TRIALS_2_LEARN, descendingExploration=False)
RUN_TYPES[DQN2L][DECISION_MAKER_NAME] = "train_dqn2l"
RUN_TYPES[DQN2L][DIRECTORY] = "trainArmy_dqn2l"
RUN_TYPES[DQN2L][HISTORY] = "replayHistory"
RUN_TYPES[DQN2L][RESULTS] = "result"


RUN_TYPES[DQN2L_DFLT] = {}
RUN_TYPES[DQN2L_DFLT][TYPE] = "DQN_WithTargetAndDefault"
RUN_TYPES[DQN2L_DFLT][PARAMS] = DQN_PARAMS_WITH_DEFAULT_DM(TRAIN_STATE.SIZE, NUM_ACTIONS, layersNum=2, numTrials2CmpResults=NUM_TRIALS_4_CMP, numTrials2Learn=NUM_TRIALS_2_LEARN, descendingExploration=False)
RUN_TYPES[DQN2L_DFLT][DECISION_MAKER_NAME] = "train_dqn2l_dflt"
RUN_TYPES[DQN2L_DFLT][DIRECTORY] = "trainArmy_dqn2l_dflt"
RUN_TYPES[DQN2L_DFLT][HISTORY] = "replayHistory"
RUN_TYPES[DQN2L_DFLT][RESULTS] = "result"

RUN_TYPES[A2C] = {}
RUN_TYPES[A2C][TYPE] = "A2C"
RUN_TYPES[A2C][PARAMS] = A2C_PARAMS(TRAIN_STATE.SIZE, NUM_ACTIONS, numTrials2CmpResults=NUM_TRIALS_4_CMP, numTrials2Learn=NUM_TRIALS_2_LEARN)
RUN_TYPES[A2C][DECISION_MAKER_NAME] = "train_A2C"
RUN_TYPES[A2C][DIRECTORY] = "trainArmy_A2C"
RUN_TYPES[A2C][HISTORY] = "replayHistory"
RUN_TYPES[A2C][RESULTS] = "results"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "trainArmy_naive"
RUN_TYPES[NAIVE][RESULTS] = "trainArmy_result"


def CreateDecisionMakerTrainArmy(configDict, isMultiThreaded, dmCopy=None):
    dmCopy = "" if dmCopy==None else "_" + str(dmCopy)

    if configDict[AGENT_NAME] == "none":
        return BaseDecisionMaker(AGENT_NAME), []
    
    runType = RUN_TYPES[configDict[AGENT_NAME]]
    # create agent dir
    directory = configDict["directory"] + "/" + AGENT_DIR

    if configDict[AGENT_NAME] == "naive":
        decisionMaker = NaiveDecisionMakerTrain(resultFName=runType[RESULTS], directory=directory + runType[DIRECTORY] + dmCopy)
    else:
        if runType[TYPE] == "DQN_WithTargetAndDefault":
            runType[PARAMS].defaultDecisionMaker = NaiveDecisionMakerTrain()

        decisionMaker = DecisionMakerExperienceReplay(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  agentName=AGENT_NAME, 
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory + runType[DIRECTORY] + dmCopy, isMultiThreaded=isMultiThreaded)
    return decisionMaker, runType

class NaiveDecisionMakerTrain(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Save=NUM_TRIALS_2_LEARN):
        super(NaiveDecisionMakerTrain, self).__init__(numTrials2Save=numTrials2Save, resultFName=resultFName, directory=directory, agentName=AGENT_NAME)

    def ActionsValues(self, state, validActions, target = True):  
        vals = np.zeros(NUM_ACTIONS,dtype = float)
           
        if state[TRAIN_STATE.ARMY_POWER] < 20:
            vals[ID_ACTION_TRAIN_MARINE] = 0.4
            vals[ID_ACTION_TRAIN_REAPER] = 0.4
            vals[ID_ACTION_TRAIN_SIEGETANK] = 0.5
            vals[ID_ACTION_TRAIN_HELLION] = 0.5                    
        elif state[TRAIN_STATE.ARMY_POWER] < 30:
            vals[ID_ACTION_TRAIN_MARINE] = 0.1
            vals[ID_ACTION_TRAIN_REAPER] = 0.1
            vals[ID_ACTION_TRAIN_SIEGETANK] = 0.5
            vals[ID_ACTION_TRAIN_HELLION] = 0.4 
        else: 
            vals[ID_ACTION_TRAIN_SIEGETANK] = 0.5
            vals[ID_ACTION_TRAIN_HELLION] = 0.4 

        if state[TRAIN_STATE.BARRACKS_IDX] == 0:
            vals[ID_ACTION_TRAIN_MARINE] = -0.1
            vals[ID_ACTION_TRAIN_REAPER] = -0.1

        if state[TRAIN_STATE.FACTORY_IDX] == 0:
            vals[ID_ACTION_TRAIN_HELLION] = -0.1
            vals[ID_ACTION_TRAIN_SIEGETANK] = -0.1  
        
        if state[TRAIN_STATE.TECHLAB_IDX] == 0:
            vals[ID_ACTION_TRAIN_SIEGETANK] = -0.1  

        vals[ID_ACTION_DO_NOTHING] = 0.1

        return vals

    def choose_action(self, state, validActions, targetValues=False):
        action = ID_ACTION_DO_NOTHING
        
        if ID_ACTION_TRAIN_SIEGETANK in validActions:
            action = ID_ACTION_TRAIN_SIEGETANK
        elif ID_ACTION_TRAIN_HELLION in validActions:
            action = ID_ACTION_TRAIN_HELLION
        else:
            if np.random.uniform() > 0.5:
                if ID_ACTION_TRAIN_MARINE in validActions:
                    action = ID_ACTION_TRAIN_MARINE
                elif ID_ACTION_TRAIN_REAPER in validActions:
                    action = ID_ACTION_TRAIN_REAPER
            else:
                if ID_ACTION_TRAIN_MARINE in validActions:
                    action = ID_ACTION_TRAIN_MARINE
                elif ID_ACTION_TRAIN_REAPER in validActions:
                    action = ID_ACTION_TRAIN_REAPER

        return action       


class TrainArmySubAgent(BaseAgent):
    def __init__(self, sharedData, configDict, decisionMaker, isMultiThreaded, playList, trainList, testList, dmCopy=None):     
        super(TrainArmySubAgent, self).__init__(TRAIN_STATE.SIZE)     

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList
        self.inTraining = self.trainAgent
        
        self.illigalmoveSolveInModel = True

        # tables:
        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker, _ = CreateDecisionMakerTrainArmy(configDict, isMultiThreaded)

        self.history = self.decisionMaker.AddHistory()
        
        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.sharedData = sharedData

        # model params
        self.actionsRequirement = {}
        self.actionsRequirement[ID_ACTION_TRAIN_MARINE] = ActionRequirement(50, 0, Terran.Barracks)
        self.actionsRequirement[ID_ACTION_TRAIN_REAPER] = ActionRequirement(50, 50, Terran.Barracks)
        self.actionsRequirement[ID_ACTION_TRAIN_HELLION] = ActionRequirement(100, 0, Terran.Factory)
        self.actionsRequirement[ID_ACTION_TRAIN_SIEGETANK] = ActionRequirement(150, 125, Terran.FactoryTechLab)

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
        super(TrainArmySubAgent, self).FirstStep(obs)   

        # states and action:
        self.current_action = None
        self.current_state = np.zeros(TRAIN_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(TRAIN_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(TRAIN_STATE.SIZE, dtype=np.int32, order='C')
        
        self.lastActionCommittedStep = 0
        self.lastActionCommittedState = None
        self.lastActionCommittedNextState = None

        self.countBuildingsQueue = {}
        for key in TRAIN_STATE.BUILDING_2_STATE_QUEUE_TRANSITION.keys():
            self.countBuildingsQueue = []

    def IsDoNothingAction(self, a):
        return a == ID_ACTION_DO_NOTHING

    def EndRun(self, reward, score, stepNum):
        saved = False
        if self.trainAgent or self.testAgent:
            saved = self.decisionMaker.end_run(reward, score, stepNum)

        return saved


    def Action2SC2Action(self, obs, a, moveNum):
        if moveNum == 0:
            finishedAction = False
            if a == None:
                print("ERROR\n\n None\n\n")
            self.buildingSelected = self.SelectBuilding2Train(a)
            if self.buildingSelected != None:
                target = self.buildingSelected.m_screenLocation
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), finishedAction
        
        elif moveNum == 1:
            if 'build_queue' in obs.observation:
                self.CorrectQueue(obs.observation['build_queue'])

            finishedAction = True
            unit2Train = ACTION_2_UNIT[a]
            sc2Action = TerranUnit.ARMY_SPEC[unit2Train].sc2Action
            if sc2Action in obs.observation['available_actions']:
                self.sharedData.prevTrainActionReward = self.sharedData.unitTrainValue[unit2Train]
                self.Add2Q(a)
                return actions.FunctionCall(sc2Action, [SC2_Params.QUEUED]), finishedAction

        return DO_NOTHING_SC2_ACTION, True

    def Learn(self, reward, terminal):        
        if self.history != None and self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)

            if self.isActionCommitted:
                self.history.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            
            elif reward > 0:
                if self.lastActionCommitted != None:
                    numSteps = self.lastActionCommittedStep - self.sharedData.numAgentStep
                    discountedReward = reward * pow(self.decisionMaker.DiscountFactor(), numSteps)
                    self.history.learn(self.lastActionCommittedState, self.lastActionCommitted, discountedReward, self.lastActionCommittedNextState, terminal)  

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def CreateState(self, obs):     

        self.current_state[TRAIN_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[TRAIN_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]
        
        for key, value in TRAIN_STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value] = self.NumBuildings(key)

        supplyUsed = obs.observation['player'][SC2_Params.SUPPLY_USED]
        supplyCap = obs.observation['player'][SC2_Params.SUPPLY_CAP]
        self.current_state[TRAIN_STATE.SUPPLY_LEFT] = supplyCap - supplyUsed

        power = 0.0
        for unit, num in self.sharedData.armySize.items():
            power += num * self.sharedData.unitTrainValue[unit]

        self.current_state[TRAIN_STATE.ARMY_POWER] = int(power)

        self.sharedData.qMinSizes = self.CalculateQueues()
        for key, value in TRAIN_STATE.BUILDING_2_STATE_QUEUE_TRANSITION.items():
            if key in self.sharedData.qMinSizes:
                self.current_state[value] = self.sharedData.qMinSizes[key]
            else:
                self.current_state[value] = TRAIN_STATE.NON_EXIST_Q_VAL

        self.current_state[TRAIN_STATE.TIME_LINE_IDX] = self.sharedData.numStep 

        self.ScaleState()

        if self.isActionCommitted:
            self.lastActionCommittedStep = self.sharedData.numAgentStep
            self.lastActionCommittedState = self.previous_scaled_state
            self.lastActionCommittedNextState = self.current_scaled_state

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]

        self.current_scaled_state[TRAIN_STATE.MINERALS_IDX] = min(TRAIN_STATE.MINERALS_MAX, self.current_scaled_state[TRAIN_STATE.MINERALS_IDX])
        self.current_scaled_state[TRAIN_STATE.GAS_IDX] = min(TRAIN_STATE.GAS_MAX, self.current_scaled_state[TRAIN_STATE.GAS_IDX])

        self.current_scaled_state[TRAIN_STATE.SUPPLY_LEFT] = min(TRAIN_STATE.SUPPLY_LEFT_MAX, self.current_scaled_state[TRAIN_STATE.SUPPLY_LEFT])

    def NumBuildings(self, buildingType):
        num = len(self.sharedData.buildingCompleted[buildingType])
        if buildingType in TRAIN_STATE.BUILDING_2_ADDITION:
            num += len(self.sharedData.buildingCompleted[TRAIN_STATE.BUILDING_2_ADDITION[buildingType]])

        return num

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions(self.current_scaled_state)
            else: 
                validActions = list(range(NUM_ACTIONS))
            
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)
        else:
            action = self.subAgentPlay

        self.current_action = action
        return action

    def ValidActions(self, state):
        valid = [ID_ACTION_DO_NOTHING]
        for key, requirement in self.actionsRequirement.items():
            if self.ValidSingleAction(state, requirement):
                valid.append(key)
        return valid

    def ValidSingleAction(self, state, requirement):
        hasMinerals = state[TRAIN_STATE.MINERALS_IDX] >= requirement.mineralsPrice
        hasGas = state[TRAIN_STATE.GAS_IDX] >= requirement.gasPrice
        idx = TRAIN_STATE.BUILDING_2_STATE_TRANSITION[requirement.buildingDependency]
        otherReq = state[idx] > 0
        return hasMinerals & hasGas & otherReq

    def Action2Str(self, a, onlyAgent=False):
        return ACTION2STR[a]

    def StateIdx2Str(self, idx):
        return TRAIN_STATE.IDX2STR[idx]

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


    def CorrectQueue(self, buildingQueue):
        stepCompleted = -1000
        self.buildingSelected.qForProduction = []
        for bq in buildingQueue:
            unit = TrainCmd(bq[SC2_Params.UNIT_TYPE_IDX])
            
            if bq[SC2_Params.COMPLETION_RATIO_IDX] > 0:
                unit.step = bq[SC2_Params.COMPLETION_RATIO_IDX]
            else:
                unit.step = stepCompleted
                stepCompleted -= 1

            self.buildingSelected.qForProduction.append(unit)

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