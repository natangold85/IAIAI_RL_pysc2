# build base sub agent
import sys
import random
import math
import time
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions

import tensorflow as tf

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_tables import TableMngr
from utils_tables import TestTableMngr
from utils_tables import LearnWithReplayMngr
from utils_tables import QLearningTable
from utils_tables import ResultFile
from utils_tables import UserPlay

from utils_tables import DQN_PARAMS
from utils_tables import DQN_EMBEDDING_PARAMS
from utils_tables import QTableParams
from utils_tables import QTableParamsExplorationDecay

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp
from utils import CenterPoints

TIME_LINE_BUCKETING = 25
NON_VALID_LOCATION_QVAL = -2

def NumActions(gridSize):
    return gridSize * gridSize + 1
def NumStateLocationsVal(gridSize):
    return 3 * gridSize * gridSize
def NumStateVal(gridSize):
    return 3 * gridSize * gridSize + 1


#physical actions from q table
ACTION_DO_NOTHING = 0
ACTION_NORTH = 1
ACTION_SOUTH = 2
ACTION_EAST = 3
ACTION_WEST = 4
ACTION_NORTH_EAST = 5
ACTION_NORTH_WEST = 6
ACTION_SOUTH_EAST = 7
ACTION_SOUTH_WEST = 8

NUM_PHYSICAL_ACTIONS =  9

NUM_UNIT_SCREEN_PIXELS = 0

SCREEN_MIN = [3,3]
SCREEN_MAX = [59,80]

for key,value in TerranUnit.UNIT_SPEC.items():
    if value.name == "marine":
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels

# possible types of play

DQN_GS5_ARMY_ATTACK_NOILLIGAL = 'dqnGS5_ArmyAttack_NoIlligal'
DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING = 'dqnGS5_ArmyAttack_NoIlligalDoNothing'
DQN_GS5_BASE_ATTACK_NOILLIGAL = 'dqnGS5_BaseAttack_NoIlligal'
DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS = 'dqnGS5_DirectBase_buildingWeights'
DQN_GS5_EMBEDDING_LOCATIONS = 'dqnGS5_ArmyAttack_NoIlligalDoNothing_Embedding' 

DQN_GS5_BASE = 'dqnGS5_BaseAttack'
DQN_GS5_BASE_EMBEDDING_LOCATIONS = 'dqnGS5_BaseAttack_Embedding' 
DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE = 'dqnGS5_BaseAttack_EmbeddingReuse' 

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS, DQN_GS5_ARMY_ATTACK_NOILLIGAL, DQN_GS5_BASE_ATTACK_NOILLIGAL, 
            DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING, DQN_GS5_EMBEDDING_LOCATIONS,
            DQN_GS5_BASE, DQN_GS5_BASE_EMBEDDING_LOCATIONS, DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE])

def BaseTestSet(gridSize):    
    testSet = []

    s1 = np.zeros(3 * gridSize * gridSize + 1, dtype = int)
    
    s1[8] = 5
    s1[gridSize * gridSize + 8] = 1
    s1[gridSize * gridSize + 9] = 1
    s1[gridSize * gridSize + 13] = 1
    s1[2 * gridSize * gridSize + 14] = 1
    actions_s1 = [8, 9, 13, 14, gridSize * gridSize]

    testSet.append([s1, actions_s1])

    s2 = np.zeros(3 * gridSize * gridSize + 1, dtype = int)

    s2[5] = 2
    s2[gridSize * gridSize + 8] = 2
    s2[gridSize * gridSize + 14] = 3
    actions_s2 = [8, 14, gridSize * gridSize]

    testSet.append([s2, actions_s2])

    s3 = np.zeros(3 * gridSize * gridSize + 1, dtype = int)
    s3[11] = 5
    s3[gridSize * gridSize + 12] = 2
    s3[gridSize * gridSize + 23] = 2
    s3[2 * gridSize * gridSize + 13] = 1
    actions_s3 = [12, 13, 23, gridSize * gridSize]

    testSet.append([s3, actions_s3])

    return testSet

# table type
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

# table names
RUN_TYPES = {}

def build_nn_sig(x, numActions, scope = 'dqn', nameChar = 'l'):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512)
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
    return output

def build_nn_processedState(x, numActions, scope = 'dqn', nameChar = 'l'):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512)
        output = tf.contrib.layers.fully_connected(fc1, numActions)
    return output

    # Define the neural network
def build_dqn_0init(x, numActions, scope):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512, activation_fn = tf.nn.softplus, weights_initializer=tf.zeros_initializer())
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid, weights_initializer=tf.zeros_initializer()) * 2 - 1
    return output

RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL] = {}
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][TYPE] = "NN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][NN] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal_DQN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][HISTORY] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal_replayHistory"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][RESULTS] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal_result"

RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING] = {}
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][TYPE] = "NN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][NN] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_DQN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][HISTORY] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_replayHistory"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_DONOTHING][RESULTS] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_result"

RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][TYPE] = "NN"
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(NumStateVal(5), NumStateLocationsVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][NN] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_Embedding_DQN"
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][HISTORY] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_Embedding_replayHistory"
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][RESULTS] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_Embedding_result"

RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL] = {}
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][TYPE] = "NN"
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][NN] = "battleMngr_dqnGS5_BaseAttack_NoIlligal_DQN"
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][HISTORY] = "battleMngr_dqnGS5_BaseAttack_NoIlligal_replayHistory"
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][RESULTS] = "battleMngr_dqnGS5_BaseAttack_NoIlligal_result"



RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS] = {}
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][TYPE] = "NN"
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][BUILDING_VALUES_key] = True
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][NN] = "battleMngr_dqnGS5_DirectBaseAttack_BuildingWeights_DQN"
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][HISTORY] = "battleMngr_dqnGS5_DirectBaseAttack_BuildingWeights_replayHistory"
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][RESULTS] = "battleMngr_dqnGS5_DirectBaseAttack_BuildingWeights_result"

RUN_TYPES[DQN_GS5_BASE] = {}
RUN_TYPES[DQN_GS5_BASE][TYPE] = "NN"
RUN_TYPES[DQN_GS5_BASE][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_BASE][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_BASE][NN] = "battleMngr_dqnGS5_BaseAttack_DQN"
RUN_TYPES[DQN_GS5_BASE][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_BASE][HISTORY] = "battleMngr_dqnGS5_BaseAttack_replayHistory"
RUN_TYPES[DQN_GS5_BASE][RESULTS] = "battleMngr_dqnGS5_BaseAttack_result"


RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][TYPE] = "NN"
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(NumStateVal(5), NumStateLocationsVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][NN] = "battleMngr_dqnGS5_BaseAttack_Embedding_DQN"
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][HISTORY] = "battleMngr_dqnGS5_BaseAttack_Embedding_replayHistory"
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS][RESULTS] = "battleMngr_dqnGS5_BaseAttack_Embedding_result"

# trained network after 15400 games in armyattack
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE] = {}
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][TYPE] = "NN"
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][PARAMS] = DQN_EMBEDDING_PARAMS(NumStateVal(5), NumStateLocationsVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][NN] = "battleMngr_dqnGS5_ArmyAttack_NoIlligalDoNothing_Embedding_DQN"
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][HISTORY] = "battleMngr_dqnGS5_BaseAttack_EmbeddingReuse_replayHistory"
RUN_TYPES[DQN_GS5_BASE_EMBEDDING_LOCATIONS_REUSE][RESULTS] = "battleMngr_dqnGS5_BaseAttack_EmbeddingReuse_result"


RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][GRIDSIZE_key] = 5
RUN_TYPES[USER_PLAY][Q_TABLE_INPUT] = "meleeAgent_qGS5_qtable"
RUN_TYPES[USER_PLAY][TYPE] = "play"
    
DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

STEP_DURATION = 0

#Local actions coordinates
TO_MOVE = 4
GOTO_CHANGE = {}
GOTO_CHANGE[ACTION_NORTH] = [-TO_MOVE , 0]
GOTO_CHANGE[ACTION_SOUTH] = [TO_MOVE , 0]
GOTO_CHANGE[ACTION_EAST] = [0 , TO_MOVE]
GOTO_CHANGE[ACTION_WEST] = [0 , -TO_MOVE]
GOTO_CHANGE[ACTION_NORTH_EAST] = [-TO_MOVE , TO_MOVE]
GOTO_CHANGE[ACTION_NORTH_WEST] = [-TO_MOVE , -TO_MOVE]
GOTO_CHANGE[ACTION_SOUTH_EAST] = [TO_MOVE , TO_MOVE]
GOTO_CHANGE[ACTION_SOUTH_WEST] = [TO_MOVE , -TO_MOVE]



class Attack(base_agent.BaseAgent):
    def __init__(self):        
        super(Attack, self).__init__()

        runTypeArg = ALL_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("\n\nplay type not entered correctly\n\n")
            exit() 

        runArg = runTypeArg.pop()
        runType = RUN_TYPES[runArg]

        # state and actions:

        self.current_action = None
        self.gridSize = runType[GRIDSIZE_key]
        self.numActions = self.gridSize * self.gridSize + 1

        self.state_startSelfMat = 0
        self.state_startEnemyMat = self.gridSize * self.gridSize
        self.state_startBuildingMat = 2 * self.gridSize * self.gridSize
        self.state_timeLineIdx = 3 * self.gridSize * self.gridSize

        self.state_size = 3 * self.gridSize * self.gridSize + 1
        self.terminalStates = self.TerminalStates()
        
        self.doNtohingStop = True
        self.illigalmoveSolveInModel = True
        self.illigalMoveReward = 0
        # if runArg.find("NoIlligal") >= 0:
        #     self.illigalMoveReward = 0
        #     self.illigalmoveSolveInModel = True
        # else:
        #     self.illigalMoveReward = -1.0
        #     self.illigalmoveSolveInModel = False

        # if runArg.find("DoNothing") >= 0:
        #     self.doNtohingStop = True
        # else:
        #     self.doNtohingStop = False

        self.BuildingValues = {}
        if BUILDING_VALUES_key in runType.keys():
            for spec in TerranUnit.BUILDING_SPEC.values():
                self.BuildingValues[spec.name] = spec.buildingValues
        else:
            for spec in TerranUnit.BUILDING_SPEC.values():
                self.BuildingValues[spec.name] = 1

        # create decision maker
        if runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            runType[PARAMS].stateSize = self.state_size
            self.decisionMaker = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            self.doNtohingAction = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])
            self.decisionMaker = UserPlay(False)
        else:
            print("\n\ninvalid run type\n\n")
            exit()

        self.lastValidAttackAction = None
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.enemyBuildingGridLoc2ScreenLoc = {}


    def step(self, obs):
        super(Attack, self).step(obs)
        if obs.first():
            return self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
            
        self.sumReward += obs.reward
        self.CreateState(obs)
        if self.doNtohingStop and SC2_Actions.STOP in obs.observation['available_actions']:
            sc2Action = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])
        else:
            sc2Action = DO_NOTHING_SC2_ACTION
        if self.errorOccur:
            return sc2Action
        # print("\n")
        # self.PrintState()
        # print(self.enemyArmyGridLoc2ScreenLoc)
        # print(self.enemyBuildingGridLoc2ScreenLoc)

        self.numStep += 1
        self.Learn()
        
        self.current_action = self.ChooseAction()
        time.sleep(STEP_DURATION)

        if self.current_action < self.gridSize * self.gridSize:
            validAction = False
            if self.current_action in self.enemyArmyGridLoc2ScreenLoc.keys():   
                goTo = self.enemyArmyGridLoc2ScreenLoc[self.current_action].copy()
                validAction = True
            elif self.current_action in self.enemyBuildingGridLoc2ScreenLoc.keys(): 
                goTo = self.enemyBuildingGridLoc2ScreenLoc[self.current_action].copy()
                validAction = True  
                
            if validAction:
                if SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                    self.lastValidAttackAction = self.current_action
                    sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
            else:
                self.prevReward += self.illigalMoveReward

                
        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0

        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.current_action = None
        self.lastValidAttackAction = self.gridSize * self.gridSize
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.enemyBuildingGridLoc2ScreenLoc = {} 
        self.selfLocCoord = None      
        self.errorOccur = False

        self.sumReward = 0
        self.prevReward = 0

        return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])

    def LastStep(self, obs):
        if obs.reward > 0:
            reward = 1
            s_ = self.terminalStates["win"]
        elif obs.reward < 0:
            reward = -1
            s_ = self.terminalStates["loss"]
        else:
            reward = -1
            s_ = self.terminalStates["tie"]

        if self.current_action is not None:
            self.decisionMaker.learn(self.current_state.copy(), self.current_action, float(reward), s_, True)

        score = obs.observation["score_cumulative"][0]
        self.decisionMaker.end_run(reward, score, self.numStep)
    
    def Learn(self):
        if self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, self.prevReward, self.current_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.prevReward = 0

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

    def CreateState(self, obs):
        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyArmyLoc(obs)
        self.GetEnemyBuildingLoc(obs)
        
        self.current_state[self.state_timeLineIdx] = int(self.numStep / TIME_LINE_BUCKETING)

    def GetSelfLoc(self, obs):
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        s_y, s_x = (screenMap == SC2_Params.PLAYER_SELF).nonzero()
        selfPoints, selfPower = CenterPoints(s_y, s_x)

        for i in range(len(selfPoints)):
            idx = self.GetScaledIdx(selfPoints[i])
            power = math.ceil(selfPower[i] /  NUM_UNIT_SCREEN_PIXELS)
            self.current_state[self.state_startSelfMat + idx] += power

        if len(s_y) > 0:
            self.selfLocCoord = [int(sum(s_y) / len(s_y)), int(sum(s_x) / len(s_x))]

    def GetEnemyArmyLoc(self, obs):
        playerType = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["screen"][SC2_Params.UNIT_TYPE]

        enemyPoints = []
        enemyPower = []
        for unit, spec in TerranUnit.UNIT_SPEC.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            unitPoints, unitPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
            enemyPoints += unitPoints
            enemyPower += unitPower
            
        self.enemyArmyGridLoc2ScreenLoc = {}
        for i in range(len(enemyPoints)):
            idx = self.GetScaledIdx(enemyPoints[i])
            if idx in self.enemyArmyGridLoc2ScreenLoc.keys():
                self.current_state[self.state_startEnemyMat + idx] += enemyPower[i]
                self.enemyArmyGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyArmyGridLoc2ScreenLoc[idx], enemyPoints[i])
            else:
                self.current_state[self.state_startEnemyMat + idx] = enemyPower[i]
                self.enemyArmyGridLoc2ScreenLoc[idx] = enemyPoints[i]

    def GetEnemyBuildingLoc(self, obs):
        playerType = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["screen"][SC2_Params.UNIT_TYPE]

        enemyBuildingPoints = []
        enemyBuildingPower = []
        for unit, spec in TerranUnit.BUILDING_SPEC.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            buildingPoints, buildingPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
            enemyBuildingPoints += buildingPoints
            enemyBuildingPower += buildingPower * self.BuildingValues[spec.name]
        
        self.enemyBuildingGridLoc2ScreenLoc = {}
        for i in range(len(enemyBuildingPoints)):
            idx = self.GetScaledIdx(enemyBuildingPoints[i])
            if idx in self.enemyBuildingGridLoc2ScreenLoc.keys():
                self.current_state[self.state_startBuildingMat + idx] += enemyBuildingPower[i]
                self.enemyBuildingGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyBuildingGridLoc2ScreenLoc[idx], enemyBuildingPoints[i])
            else:
                self.current_state[self.state_startBuildingMat + idx] = enemyBuildingPower[i]
                self.enemyBuildingGridLoc2ScreenLoc[idx] = enemyBuildingPoints[i]        
       
    def GetScaledIdx(self, screenCord):
        locX = screenCord[SC2_Params.X_IDX]
        locY = screenCord[SC2_Params.Y_IDX]

        yScaled = int((locY / SC2_Params.SCREEN_SIZE) * self.gridSize)
        xScaled = int((locX / SC2_Params.SCREEN_SIZE) * self.gridSize)

        return xScaled + yScaled * self.gridSize
    
    def Closest2Self(self, p1, p2):
        d1 = DistForCmp(p1, self.selfLocCoord)
        d2 = DistForCmp(p2, self.selfLocCoord)
        if d1 < d2:
            return p1
        else:
            return p2
    
    def ValidActions(self):
        locEnemy = list(self.enemyArmyGridLoc2ScreenLoc.keys())
        locBuildings = list(self.enemyBuildingGridLoc2ScreenLoc.keys())
            
        return list(set(locEnemy + locBuildings + [self.gridSize * self.gridSize]))

    def TerminalStates(self):
        tStates = {}

        state = np.zeros(self.state_size, dtype=np.int32, order='C')
        state[0] = -1
        tStates["win"] = state.copy()
        state[0] = -2
        tStates["tie"] = state.copy()
        state[0] = -3
        tStates["loss"] = state.copy()

        return tStates

    def PrintState(self):
        print("\n\nstate: timeline =", self.current_state[self.state_timeLineIdx], "last attack action =", self.lastValidAttackAction)
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                idx = self.state_startSelfMat + x + y * self.gridSize
                print(int(self.current_state[idx]), end = '')
            
            print(end = '  |  ')
            
            for x in range(self.gridSize):
                idx = self.state_startEnemyMat + x + y * self.gridSize
                print(int(self.current_state[idx]), end = '')

            print(end = '  |  ')
            
            for x in range(self.gridSize):
                idx = self.state_startBuildingMat + x + y * self.gridSize
                if self.current_state[idx] < 10:
                    print(self.current_state[idx], end = '  ')
                else:
                    print(self.current_state[idx], end = ' ')

            print('||')

