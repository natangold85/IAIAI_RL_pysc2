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

from algo_decisionMaker import CreateDecisionMaker

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

AGENT_NAME = "train_army"


# Model Params
NUM_TRIALS_2_LEARN = 20
NUM_TRIALS_4_CMP = 200

class TRAIN_ACTIONS:
    DO_NOTHING = 0
    TRAIN_MARINE = 1
    TRAIN_REAPER = 2
    TRAIN_HELLION = 3
    TRAIN_SIEGETANK = 4

    SIZE = 5

ACTION2STR = ["DoNothing" , "TrainMarine", "TrainReaper", "TrainHellion", "TrainSiegeTank"]

ACTION_2_UNIT = {}
ACTION_2_UNIT[TRAIN_ACTIONS.TRAIN_MARINE] = GetUnitId("marine")
ACTION_2_UNIT[TRAIN_ACTIONS.TRAIN_REAPER] = GetUnitId("reaper")
ACTION_2_UNIT[TRAIN_ACTIONS.TRAIN_HELLION] = GetUnitId("hellion")
ACTION_2_UNIT[TRAIN_ACTIONS.TRAIN_SIEGETANK] = GetUnitId("siege tank")

ACTION_2_BUILDING = {}
ACTION_2_BUILDING[TRAIN_ACTIONS.DO_NOTHING] = None
ACTION_2_BUILDING[TRAIN_ACTIONS.TRAIN_MARINE] = Terran.Barracks
ACTION_2_BUILDING[TRAIN_ACTIONS.TRAIN_REAPER] = Terran.Barracks
ACTION_2_BUILDING[TRAIN_ACTIONS.TRAIN_HELLION] = Terran.Factory
ACTION_2_BUILDING[TRAIN_ACTIONS.TRAIN_SIEGETANK] = Terran.Factory

ACTION_2_ADDITION = {}
ACTION_2_ADDITION[TRAIN_ACTIONS.TRAIN_SIEGETANK] = Terran.FactoryTechLab

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

class NaiveDecisionMakerTrain(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Save=100):
        super(NaiveDecisionMakerTrain, self).__init__(numTrials2Save=numTrials2Save, resultFName=resultFName, directory=directory, agentName=AGENT_NAME)

    def ActionsValues(self, state, validActions, target = True):  
        vals = np.zeros(TRAIN_ACTIONS.SIZE,dtype = float)
           
        if state[TRAIN_STATE.ARMY_POWER] < 20:
            vals[TRAIN_ACTIONS.TRAIN_MARINE] = 0.4
            vals[TRAIN_ACTIONS.TRAIN_REAPER] = 0.4
            vals[TRAIN_ACTIONS.TRAIN_SIEGETANK] = 0.5
            vals[TRAIN_ACTIONS.TRAIN_HELLION] = 0.5                    
        elif state[TRAIN_STATE.ARMY_POWER] < 30:
            vals[TRAIN_ACTIONS.TRAIN_MARINE] = 0.1
            vals[TRAIN_ACTIONS.TRAIN_REAPER] = 0.1
            vals[TRAIN_ACTIONS.TRAIN_SIEGETANK] = 0.5
            vals[TRAIN_ACTIONS.TRAIN_HELLION] = 0.4 
        else: 
            vals[TRAIN_ACTIONS.TRAIN_SIEGETANK] = 0.5
            vals[TRAIN_ACTIONS.TRAIN_HELLION] = 0.4 

        if state[TRAIN_STATE.BARRACKS_IDX] == 0:
            vals[TRAIN_ACTIONS.TRAIN_MARINE] = -0.1
            vals[TRAIN_ACTIONS.TRAIN_REAPER] = -0.1

        if state[TRAIN_STATE.FACTORY_IDX] == 0:
            vals[TRAIN_ACTIONS.TRAIN_HELLION] = -0.1
            vals[TRAIN_ACTIONS.TRAIN_SIEGETANK] = -0.1  
        
        if state[TRAIN_STATE.TECHLAB_IDX] == 0:
            vals[TRAIN_ACTIONS.TRAIN_SIEGETANK] = -0.1  

        vals[TRAIN_ACTIONS.DO_NOTHING] = 0.1

        return vals

    def choose_action(self, state, validActions, targetValues=False):
        action = TRAIN_ACTIONS.DO_NOTHING
        
        if TRAIN_ACTIONS.TRAIN_SIEGETANK in validActions:
            action = TRAIN_ACTIONS.TRAIN_SIEGETANK
        elif TRAIN_ACTIONS.TRAIN_HELLION in validActions:
            action = TRAIN_ACTIONS.TRAIN_HELLION
        else:
            if np.random.uniform() > 0.5:
                if TRAIN_ACTIONS.TRAIN_MARINE in validActions:
                    action = TRAIN_ACTIONS.TRAIN_MARINE
                elif TRAIN_ACTIONS.TRAIN_REAPER in validActions:
                    action = TRAIN_ACTIONS.TRAIN_REAPER
            else:
                if TRAIN_ACTIONS.TRAIN_MARINE in validActions:
                    action = TRAIN_ACTIONS.TRAIN_MARINE
                elif TRAIN_ACTIONS.TRAIN_REAPER in validActions:
                    action = TRAIN_ACTIONS.TRAIN_REAPER

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
            self.decisionMaker, _ = CreateDecisionMaker(agentName=AGENT_NAME, configDict=configDict,  
                                                    isMultiThreaded=isMultiThreaded, dmCopy=dmCopy, heuristicClass=NaiveDecisionMakerTrain)

        self.history = self.decisionMaker.AddHistory()
        
        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.sharedData = sharedData

        # model params
        self.actionsRequirement = {}
        self.actionsRequirement[TRAIN_ACTIONS.TRAIN_MARINE] = ActionRequirement(50, 0, Terran.Barracks)
        self.actionsRequirement[TRAIN_ACTIONS.TRAIN_REAPER] = ActionRequirement(50, 50, Terran.Barracks)
        self.actionsRequirement[TRAIN_ACTIONS.TRAIN_HELLION] = ActionRequirement(100, 0, Terran.Factory)
        self.actionsRequirement[TRAIN_ACTIONS.TRAIN_SIEGETANK] = ActionRequirement(150, 125, Terran.FactoryTechLab)

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
        return a == TRAIN_ACTIONS.DO_NOTHING

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
        
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

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
                validActions = list(range(TRAIN_ACTIONS.SIZE))
            
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)
        else:
            action = self.subAgentPlay

        self.current_action = action
        return action

    def ValidActions(self, state):
        valid = [TRAIN_ACTIONS.DO_NOTHING]
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
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping)