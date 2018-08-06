import random
import math
import os.path
import sys
import logging
import traceback
import datetime

import numpy as np
import pandas as pd
import time

from pysc2.lib import actions

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay

from utils_qtable import QTableParamsExplorationDecay
from utils_dqn import DQN_PARAMS

from agent_build_base import BuildBaseSubAgent
from agent_train_army import TrainArmySubAgent
from agent_do_nothing import DoNothingSubAgent

# shared data
from agent_build_base import SharedDataBuild
from agent_train_army import SharedDataTrain
from agent_resource_gather import SharedDataGather

from utils import GetScreenCorners
from utils import GetLocationForBuildingAddition
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap

AGENT_DIR = "BaseMngr/"
if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)
AGENT_NAME = "base"

ACTION_DO_NOTHING = 0
ACTION_BUILD_BASE = 1
ACTION_TRAIN_ARMY = 2
NUM_ACTIONS = 3

ACTION2STR = ["DoNothing", "BuildBase", "TrainArmy"]

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[ACTION_DO_NOTHING] = "BaseAgent"
SUBAGENTS_NAMES[ACTION_BUILD_BASE] = "BuildBaseSubAgent"
SUBAGENTS_NAMES[ACTION_TRAIN_ARMY] = "TrainArmySubAgent"

SUBAGENTS_ARGS = {}
SUBAGENTS_ARGS[ACTION_DO_NOTHING] = "naive"
SUBAGENTS_ARGS[ACTION_BUILD_BASE] = "inherit"
SUBAGENTS_ARGS[ACTION_TRAIN_ARMY] = "inherit"

class SharedDataBase(SharedDataBuild, SharedDataTrain, SharedDataGather):
    def __init__(self):
        super(SharedDataBase, self).__init__()
        self.currBaseState = None

class BASE_STATE:
    # state details
    NON_VALID_NUM = -1

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

    ARMY_POWER = 15
    QUEUE_BARRACKS = 16
    QUEUE_FACTORY = 17
    QUEUE_TECHLAB = 18

    SIZE = 19

    BUILDING_RELATED_IDX = [COMMAND_CENTER_IDX] + list(range(SUPPLY_DEPOT_IDX, ARMY_POWER))
    TRAIN_BUILDING_RELATED_IDX = [BARRACKS_IDX, FACTORY_IDX]

    BUILDING_2_STATE_TRANSITION = {}
    BUILDING_2_STATE_TRANSITION[TerranUnit.COMMANDCENTER] = [COMMAND_CENTER_IDX, -1]
    BUILDING_2_STATE_TRANSITION[TerranUnit.SUPPLY_DEPOT] = [SUPPLY_DEPOT_IDX, IN_PROGRESS_SUPPLY_DEPOT_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.OIL_REFINERY] = [REFINERY_IDX, IN_PROGRESS_REFINERY_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.BARRACKS] = [BARRACKS_IDX, IN_PROGRESS_BARRACKS_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.FACTORY] = [FACTORY_IDX, IN_PROGRESS_FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.REACTOR] = [REACTORS_IDX, IN_PROGRESS_RECTORS_IDX]
    BUILDING_2_STATE_TRANSITION[TerranUnit.TECHLAB] = [TECHLAB_IDX, IN_PROGRESS_TECHLAB_IDX]

    BUILDING_2_STATE_QUEUE_TRANSITION = {}

    BUILDING_2_STATE_QUEUE_TRANSITION[TerranUnit.BARRACKS] = QUEUE_BARRACKS
    BUILDING_2_STATE_QUEUE_TRANSITION[TerranUnit.FACTORY] = QUEUE_FACTORY
    BUILDING_2_STATE_QUEUE_TRANSITION[TerranUnit.TECHLAB] = QUEUE_TECHLAB


    IDX2STR = ["CC", "MIN", "GAS", "SD", "REF", "BA", "FA", "REA", "TECH", "SD_B", "REF_B", "BA_B", "FA_B", "REA_B", "TECH_B", "POWER", "BA_Q", "FA_Q", "TE_Q"]

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

ALL_TYPES = set([USER_PLAY, QTABLE, DQN])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(BASE_STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "baseMngr_qtable"
RUN_TYPES[QTABLE][DIRECTORY] = "baseMngr_qtable"
RUN_TYPES[QTABLE][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "baseMngr_result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(BASE_STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "baseMngr_dqn"
RUN_TYPES[DQN][DIRECTORY] = "baseMngr_dqn"
RUN_TYPES[DQN][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[DQN][RESULTS] = "baseMngr_result"

STEP_DURATION = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

NORMALIZATION = 300

UNIT_VALUE_TABLE_NAME = 'unit_value_table.gz'

NORMALIZATION_LOCAL_REWARD = 20
NORMALIZATION_GAME_REWARD = 300

class BaseMngr(BaseAgent):
    def __init__(self, runArg = None, decisionMaker = None, isMultiThreaded = False, playList = None, trainList = None):
        super(BaseMngr, self).__init__()
        
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList
        
        self.trainAgent = AGENT_NAME in trainList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(runArg, isMultiThreaded)

        # create sub agents and get decision makers
        self.subAgents = {}

        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)
            saArg = SUBAGENTS_ARGS[key]
            if saArg == "inherit":
                saArg = runArg
            
            self.subAgents[key] = saClass(saArg, saDM, isMultiThreaded, saPlayList, trainList)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())


        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.terminalState = np.zeros(BASE_STATE.SIZE, dtype=np.int, order='C')


        # model params 
        self.move_number = 0
        self.step_num = 0

        self.minPriceMinerals = 50
        self.accumulatedReward = 0

        self.unitPower = {}
        table = pd.read_pickle(UNIT_VALUE_TABLE_NAME, compression='gzip')
        valVecMarine = table.ix['marine', :]
        self.unitPower[TerranUnit.MARINE] = sum(valVecMarine) / len(valVecMarine)
        self.unitPower[TerranUnit.REAPER] = sum(table.ix['reaper', :]) / len(valVecMarine)
        self.unitPower[TerranUnit.HELLION] = sum(table.ix['hellion', :]) / len(valVecMarine)
        self.unitPower[TerranUnit.SIEGE_TANK] = sum(table.ix['siege tank', :]) / len(valVecMarine)

    def CreateDecisionMaker(self, runArg, isMultiThreaded):
        if runArg == None:
            runTypeArg = list(ALL_TYPES.intersection(sys.argv))
            runArg = runTypeArg.pop()    
        runType = RUN_TYPES[runArg]


        decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                    resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=AGENT_DIR+runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1

        for key, sa in self.subAgents.items():
            if sa.FindActingHeirarchi() >= 0:
                return key
        
        return -1

    def step(self, obs, sharedData = None, moveNum = None):
        super(BaseMngr, self).step(obs)
        self.step_num += 1
        
        try:
            self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]

            if obs.first():
                self.FirstStep(obs)

            if sharedData != None:
                self.sharedData = sharedData          
            
            if moveNum != None:
                self.moveNum = moveNum

            for sa in range(NUM_ACTIONS):
                self.subAgentsActions[sa] = self.subAgents[sa].step(obs, self.sharedData, self.move_number) 

            if self.move_number == 0:
                self.CreateState(obs)             
                self.Learn()
                self.current_action = self.ChooseAction()
                #self.PrintState()

            return self.current_action

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def FirstStep(self, obs):
        # action and state
        self.current_action = None
        
        self.previous_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')

        self.step_num = 0 
        self.move_number = 0
        self.accumulatedReward = 0

        self.sharedData = SharedDataBase()
        self.sharedData.unitTrainValue = self.unitPower

        self.subAgentsActions = {}
        for sa in range(NUM_ACTIONS):
            self.subAgentsActions[sa] = None

    def LastStep(self, obs):
        r = self.accumulatedReward / NORMALIZATION_GAME_REWARD
        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(self.current_state.copy(), self.current_action, r, self.terminalState.copy(), True)
            score = obs.observation["score_cumulative"][0]
            self.decisionMaker.end_run(r, score, self.step_num)

        for sa in range(NUM_ACTIONS):
            self.subAgents[sa].LastStep(obs, r) 
            
    def Action2SC2Action(self, obs, a, moveNum):
        return self.subAgents[a].Action2SC2Action(obs, self.subAgentsActions[a], moveNum)
    
    def IsDoNothingAction(self, a):
        return self.subAgents[a].IsDoNothingAction(self.subAgentsActions[a])
        
    def CreateState(self, obs):
        for key, value in BASE_STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.sharedData.buildingCount[key]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.sharedData.buildCommands[key])

        self.current_state[BASE_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[BASE_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        for key, value in BASE_STATE.BUILDING_2_STATE_QUEUE_TRANSITION.items():
            self.current_state[value] = len(self.sharedData.trainingQueue[key])
        
        power = 0
        for unit, num in self.sharedData.armySize.items():
            power += num * self.unitPower[unit]

        self.current_state[BASE_STATE.ARMY_POWER] = power

        self.ScaleState()

        self.sharedData.currBaseState = self.current_scaled_state.copy()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[BASE_STATE.MINERALS_IDX] = int(self.current_scaled_state[BASE_STATE.MINERALS_IDX] / BASE_STATE.MINERALS_BUCKETING) * BASE_STATE.MINERALS_BUCKETING
        self.current_scaled_state[BASE_STATE.MINERALS_IDX] = min(BASE_STATE.MINERALS_MAX, self.current_scaled_state[BASE_STATE.MINERALS_IDX])
        self.current_scaled_state[BASE_STATE.GAS_IDX] = int(self.current_scaled_state[BASE_STATE.GAS_IDX] / BASE_STATE.GAS_BUCKETING) * BASE_STATE.GAS_BUCKETING
        self.current_scaled_state[BASE_STATE.GAS_IDX] = min(BASE_STATE.GAS_MAX, self.current_scaled_state[BASE_STATE.GAS_IDX])

    def Learn(self, reward = 0):
        r = self.sharedData.prevActionReward / NORMALIZATION_LOCAL_REWARD + reward
        self.accumulatedReward += self.sharedData.prevActionReward
        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, r, self.current_state.copy())
        
        for sa in range(NUM_ACTIONS):
            self.subAgents[sa].Learn(r) 


        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.sharedData.prevActionReward = 0.0

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
        valid = [ACTION_DO_NOTHING]
        if self.current_scaled_state[BASE_STATE.MINERALS_IDX] >= self.minPriceMinerals:
            valid.append(ACTION_BUILD_BASE)
        
            if self.ArmyBuildingExist():
                valid.append(ACTION_TRAIN_ARMY)
        
        return valid

    def NewBuildings(self):
        return (self.previous_scaled_state[BASE_STATE.BUILDING_RELATED_IDX] != self.current_scaled_state[BASE_STATE.BUILDING_RELATED_IDX]).any()
    
    def ArmyBuildingExist(self):
        return (self.current_scaled_state[BASE_STATE.TRAIN_BUILDING_RELATED_IDX] > 0).any()

    def Action2Str(self, a):
        return ACTION2STR[a] + "-->" + self.subAgents[a].Action2Str(self.subAgentsActions[a])

    def PrintState(self):
        for i in range(BASE_STATE.SIZE):
            print(BASE_STATE.IDX2STR[i], self.current_scaled_state[i], end = ', ')
        print(']')
