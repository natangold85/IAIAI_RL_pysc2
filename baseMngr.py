import random
import math
import os.path
import sys
import logging
import traceback

#udp
import socket
import threading

import numpy as np
import pandas as pd
import time

from pysc2.agents import base_agent
from pysc2.lib import actions

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay

from utils_qtable import QTableParamsExplorationDecay
from utils_dqn import DQN_PARAMS

from build_base import BuildBaseSubAgent
from train_army import TrainArmySubAgent
from doNothing import DoNothingSubAgent

# shared data
from build_base import SharedDataBuild
from train_army import SharedDataTrain

from utils import GetScreenCorners
from utils import GetLocationForBuildingAddition
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap


ACTION_DO_NOTHING = 0
ACTION_BUILD_BASE = 1
ACTION_TRAIN_ARMY = 2
NUM_ACTIONS = 3
ACTION2STR = ["DoNothing", "BuildBase", "TrainArmy"]

class SharedData(SharedDataBuild, SharedDataTrain):
    def __init__(self):
        super(SharedData, self).__init__()

class STATE:
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
NN = "nn"
Q_TABLE_INPUT = "q_using"
Q_TABLE = "q"
T_TABLE = "t"
HISTORY = "hist"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'
GRIDSIZE_key = 'gridsize'
BUILDING_VALUES_key = 'buildingValues'
DIRECTORY = 'directory'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QReplay"
RUN_TYPES[QTABLE][GRIDSIZE_key] = 5
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][NN] = ""
RUN_TYPES[QTABLE][DIRECTORY] = "baseMngr_qtable"
RUN_TYPES[QTABLE][Q_TABLE] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "NN"
RUN_TYPES[DQN][GRIDSIZE_key] = 5
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][NN] = "mngr_dqn"
RUN_TYPES[DQN][DIRECTORY] = "baseMngr_dqn"
RUN_TYPES[DQN][Q_TABLE] = ""
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

STEP_DURATION = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

UNIT_2_REWARD = {}
UNIT_2_REWARD[SC2_Actions.TRAIN_MARINE] = 1
UNIT_2_REWARD[SC2_Actions.TRAIN_REAPER] = 2
UNIT_2_REWARD[SC2_Actions.TRAIN_HELLION] = 4
UNIT_2_REWARD[SC2_Actions.TRAIN_SIEGE_TANK] = 8

NORMALIZATION = 300

UNIT_VALUE_TABLE_NAME = 'unit_value_table.gz'

NORMALIZATION_LOCAL_REWARD = 50
NORMALIZATION_GAME_REWARD = 1000

class BaseMngr(base_agent.BaseAgent):
    def __init__(self):
        super(BaseMngr, self).__init__()
        
        runTypeArg = list(ALL_TYPES.intersection(sys.argv))
        runArg = runTypeArg.pop()

        runType = RUN_TYPES[runArg]

        self.illigalmoveSolveInModel = True

        # tables:
        if runType[TYPE] == "QReplay" or runType[TYPE] == "NN":
            self.decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], dqnName = runType[NN], qTableName = runType[Q_TABLE], 
                                                    resultFileName = runType[RESULTS], historyFileName=runType[HISTORY], directory=runType[DIRECTORY])
        else:
            self.decisionMaker = UserPlay(False)
        
        # sub agents
        self.subAgents = {}

        self.subAgents[ACTION_DO_NOTHING] = DoNothingSubAgent()
        self.subAgents[ACTION_BUILD_BASE] = BuildBaseSubAgent(runArg)
        self.subAgents[ACTION_TRAIN_ARMY] = TrainArmySubAgent(runArg)

        self.subAgentsActions = {}
        self.subAgentsActions[ACTION_DO_NOTHING] = 0
        self.subAgentsActions[ACTION_BUILD_BASE] = 0
        self.subAgentsActions[ACTION_TRAIN_ARMY] = 0

        self.terminalState = np.zeros(STATE.SIZE, dtype=np.int, order='C')

        # model params 
        self.move_number = 0
        self.step_num = 0

        self.minPriceMinerals = 50
        self.accumulatedReward = 0

        self.unitPower = {}
        table = pd.read_pickle(UNIT_VALUE_TABLE_NAME, compression='gzip')

        self.unitPower[TerranUnit.MARINE] = sum(table.ix['marine', :])
        self.unitPower[TerranUnit.REAPER] = sum(table.ix['reaper', :])
        self.unitPower[TerranUnit.HELLION] = sum(table.ix['hellion', :])
        self.unitPower[TerranUnit.SIEGE_TANK] = sum(table.ix['siege tank', :])

    def step(self, obs, sharedData = None):
        super(BaseMngr, self).step(obs)
        self.step_num += 1
        
        try:

            self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]

            if obs.last():
                self.LastStep(obs)
                return DO_NOTHING_SC2_ACTION
            elif obs.first():
                self.FirstStep(obs)

            if sharedData != None:
                self.sharedData = sharedData          
            
            time.sleep(STEP_DURATION)

            if self.move_number == 0:
                self.CreateState(obs)             
                self.Learn()
                self.current_action = self.ChooseAction()

                for sa in range(NUM_ACTIONS):
                    self.subAgentsActions[sa] = self.subAgents[sa].step(obs, self.sharedData, self.move_number)

                self.subAgentChosenAction = self.subAgentsActions[self.current_action]
                #self.PrintState()
            else:
                for sa in range(NUM_ACTIONS):
                    self.subAgents[sa].step(obs, self.sharedData, self.move_number)   

            #print("action =", self.Action2Str(), "move num =", self.move_number)
            return self.ActAction(obs)

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def FirstStep(self, obs):
        # action and state
        self.current_action = None
        self.subAgentChosenAction = None
        
        self.previous_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        self.step_num = 0 
        self.move_number = 0
        self.accumulatedReward = 0

        self.sharedData = SharedData()
        self.sharedData.unitTrainValue = self.unitPower

    def LastStep(self, obs):
        r = self.accumulatedReward / NORMALIZATION_GAME_REWARD
        if self.current_action is not None:
            self.decisionMaker.learn(self.current_state.copy(), self.current_action, r, self.terminalState.copy(), True)

        score = obs.observation["score_cumulative"][0]
        self.decisionMaker.end_run(r, score, self.step_num)
        for sa in range(NUM_ACTIONS):
            self.subAgents[sa].LastStep(obs, r) 

    def CreateState(self, obs):
        for key, value in STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.sharedData.buildingCount[key]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.sharedData.buildCommands[key])

        self.current_state[STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        for key, value in STATE.BUILDING_2_STATE_QUEUE_TRANSITION.items():
            self.current_state[value] = len(self.sharedData.trainingQueue[key])
        
        power = 0
        for unit, num in self.sharedData.armySize.items():
            power += num * self.unitPower[unit]

        self.current_state[STATE.ARMY_POWER] = power

        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[STATE.MINERALS_IDX] = int(self.current_scaled_state[STATE.MINERALS_IDX] / STATE.MINERALS_BUCKETING) * STATE.MINERALS_BUCKETING
        self.current_scaled_state[STATE.MINERALS_IDX] = min(STATE.MINERALS_MAX, self.current_scaled_state[STATE.MINERALS_IDX])
        self.current_scaled_state[STATE.GAS_IDX] = int(self.current_scaled_state[STATE.GAS_IDX] / STATE.GAS_BUCKETING) * STATE.GAS_BUCKETING
        self.current_scaled_state[STATE.GAS_IDX] = min(STATE.GAS_MAX, self.current_scaled_state[STATE.GAS_IDX])

    def Learn(self):
        r = self.sharedData.prevActionReward / NORMALIZATION_LOCAL_REWARD
        self.accumulatedReward += self.sharedData.prevActionReward
        if self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, r, self.current_state.copy())
            for sa in range(NUM_ACTIONS):
                self.subAgents[sa].Learn(r) 


        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.sharedData.prevActionReward = 0.0

    def ChooseAction(self):
        if self.illigalmoveSolveInModel:
            validActions = self.ValidActions()
            if np.random.uniform() > self.decisionMaker.ExploreProb():
                valVec = self.decisionMaker.actionValuesVec(self.current_state)   
                random.shuffle(validActions)
                validVal = valVec[validActions]
                action = validActions[validVal.argmax()]
            else:
                action = np.random.choice(validActions) 
        else:
            action = self.decisionMaker.choose_action(self.current_state)

        return action

    def ValidActions(self):
        valid = [ACTION_DO_NOTHING]
        if self.current_scaled_state[STATE.MINERALS_IDX] >= self.minPriceMinerals:
            valid.append(ACTION_BUILD_BASE)
        
            if self.ArmyBuildingExist():
                valid.append(ACTION_TRAIN_ARMY)
        
        return valid
    def ActAction(self, obs): 

        if self.subAgents[self.current_action].IsDoNothingAction(self.subAgentChosenAction):
            self.current_action = ACTION_DO_NOTHING
            self.subAgentChosenAction = self.subAgentsActions[self.current_action]
            sc2Action, terminal = self.subAgents[ACTION_DO_NOTHING].Action2SC2Action(obs, self.subAgentsActions[ACTION_DO_NOTHING], self.move_number)
        else: 
            sc2Action, terminal = self.subAgents[self.current_action].Action2SC2Action(obs, self.subAgentChosenAction, self.move_number)
        
        if terminal:
            self.move_number = 0
        else:
            self.move_number += 1

        return sc2Action

    def NewBuildings(self):
        return (self.previous_scaled_state[STATE.BUILDING_RELATED_IDX] != self.current_scaled_state[STATE.BUILDING_RELATED_IDX]).any()
    
    def ArmyBuildingExist(self):
        return (self.current_scaled_state[STATE.TRAIN_BUILDING_RELATED_IDX] > 0).any()

    def Action2Str(self):
        if self.subAgents[self.current_action].IsDoNothingAction(self.subAgentChosenAction):
            return ACTION2STR[ACTION_DO_NOTHING] + "-->" + self.subAgents[ACTION_DO_NOTHING].Action2Str(self.subAgentsActions[ACTION_DO_NOTHING])
        else:
            return ACTION2STR[self.current_action] + "-->" + self.subAgents[self.current_action].Action2Str(self.subAgentChosenAction)

    def PrintState(self):
        if self.current_action != None:
            print("action =", self.Action2Str(), end = "\n [ ")
        for i in range(STATE.SIZE):
            print(STATE.IDX2STR[i], self.current_scaled_state[i], end = ', ')
        print(']')
