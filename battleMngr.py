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
def NumStateVal(gridSize):
    return 2 * gridSize * gridSize + 2


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
NAIVE_MAX = "naiveMax"

DQN_SIG_GS10 = "dqnSigGS10"
DQN_SIG_GS5 = "dqnSigGS5"
QTABLE_GS10 = 'qGS10'
QTABLE_GS5 = 'qGS5'
QTABLE_GS5_DIRECT = 'qGS5_Direct'
QTABLE_GS5_DIRECT_MULTI = 'qGS5_DirectMulti'
DQN_GS5_DIRECT = 'dqnGS5_Direct'
DQN_GS5_DIRECT_MULTI = 'dqnGS5_DirectMulti'

QTABLE_GS5_DIRECT_BASE = 'qGS5_DirectBase'
DQN_GS5_DIRECT_BASE = 'dqnGS5_DirectBase'
DQN_GS5_ARMY_ATTACK = 'dqnGS5_ArmyAttack'
DQN_GS5_ARMY_ATTACK_NOILLIGAL = 'dqnGS5_ArmyAttack_NoIlligal'
DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT = 'dqnGS5_ArmyAttack_NoIlligal0Init'
DQN_GS5_BASE_ATTACK_NOILLIGAL = 'dqnGS5_BaseAttack_NoIlligal'
DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS = 'dqnGS5_DirectBase_buildingWeights'


USER_PLAY = 'play'

ATTACK_TYPES = set([USER_PLAY, QTABLE_GS10, QTABLE_GS5, DQN_SIG_GS10, DQN_SIG_GS5, NAIVE_MAX])
DIRECT_TYPES = set([USER_PLAY, QTABLE_GS5_DIRECT, DQN_GS5_DIRECT, QTABLE_GS5_DIRECT_MULTI, DQN_GS5_DIRECT_MULTI])
BASE_TYPES = set([USER_PLAY, QTABLE_GS5_DIRECT_BASE, DQN_GS5_DIRECT_BASE, DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS, DQN_GS5_ARMY_ATTACK, DQN_GS5_ARMY_ATTACK_NOILLIGAL, DQN_GS5_BASE_ATTACK_NOILLIGAL, DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT])

def BaseTestSet(gridSize):    
    
    
    s1 = np.zeros(3 * gridSize * gridSize + 2, dtype = int)
    
    s1[5] = 8
    s1[gridSize * gridSize + 8] = 2
    s1[gridSize * gridSize + 9] = 1
    s1[gridSize * gridSize + 13] = 3

    s1[2 * gridSize * gridSize + 14] = 1
    s1[2 * gridSize * gridSize + 4] = 1
    s1[3 * gridSize * gridSize] = gridSize * gridSize

    s2 = np.zeros(3 * gridSize * gridSize + 2, dtype = int)

    s2[6] = 2
    s2[gridSize * gridSize + 8] = 2
    s2[gridSize * gridSize + 9] = 3
    s2[gridSize * gridSize + 13] = 3

    s2[2 * gridSize * gridSize + 14] = 1
    s2[2 * gridSize * gridSize + 4] = 1
    s2[3 * gridSize * gridSize] = gridSize * gridSize
    
    s3 = s1.copy()
    s3[5] = 0
    s3[7] = 8
    testSet = [s1,s2,s3]
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
        fc1 = tf.contrib.layers.fully_connected(x, 512, weights_initializer=tf.zeros_initializer())
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid, weights_initializer=tf.zeros_initializer()) * 2 - 1
    return output

RUN_TYPES[DQN_SIG_GS10] = {}
RUN_TYPES[DQN_SIG_GS10][TYPE] = "NN"
RUN_TYPES[DQN_SIG_GS10][GRIDSIZE_key] = 10

RUN_TYPES[DQN_SIG_GS10][Q_TABLE_INPUT] = "meleeAgent_qGS10_qtable"
RUN_TYPES[DQN_SIG_GS10][PARAMS] = DQN_PARAMS(NumStateVal(10), NumActions(10), nn_Func = build_nn_sig)
RUN_TYPES[DQN_SIG_GS10][NN] = "battleMngr_Sig_GS10_DQN"
RUN_TYPES[DQN_SIG_GS10][Q_TABLE] = "battleMngr_Sig_GS10_qtable"
RUN_TYPES[DQN_SIG_GS10][HISTORY] = "battleMngr_Sig_GS10_historyReplay"
RUN_TYPES[DQN_SIG_GS10][RESULTS] = "battleMngr_Sig_GS10_result"


RUN_TYPES[DQN_SIG_GS5] = {}
RUN_TYPES[DQN_SIG_GS5][TYPE] = "NN"

RUN_TYPES[DQN_SIG_GS5][GRIDSIZE_key] = 5
RUN_TYPES[DQN_SIG_GS5][Q_TABLE_INPUT] = "meleeAgent_qGS5_qtable"
RUN_TYPES[DQN_SIG_GS5][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), nn_Func = build_nn_sig)
RUN_TYPES[DQN_SIG_GS5][NN] = "battleMngr_Sig_GS5_DQN"
RUN_TYPES[DQN_SIG_GS5][Q_TABLE] = "battleMngr_Sig_GS5_qtable"
RUN_TYPES[DQN_SIG_GS5][HISTORY] = "battleMngr_Sig_GS5_historyReplay"
RUN_TYPES[DQN_SIG_GS5][RESULTS] = "battleMngr_Sig_GS5_result"


RUN_TYPES[QTABLE_GS10] = {}
RUN_TYPES[QTABLE_GS10][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS10][GRIDSIZE_key] = 10
RUN_TYPES[QTABLE_GS10][Q_TABLE_INPUT] = "meleeAgent_qGS10_qtable"
RUN_TYPES[QTABLE_GS10][PARAMS] = QTableParamsExplorationDecay(NumStateVal(10), NumActions(10), propogateReward = False)
RUN_TYPES[QTABLE_GS10][NN] = ""
RUN_TYPES[QTABLE_GS10][Q_TABLE] = "battleMngr_qGS10_qtable"
RUN_TYPES[QTABLE_GS10][HISTORY] = "battleMngr_qGS10_replayHistory"
RUN_TYPES[QTABLE_GS10][RESULTS] = "battleMngr_qGS10_result"

RUN_TYPES[QTABLE_GS5] = {}
RUN_TYPES[QTABLE_GS5][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS5][GRIDSIZE_key] = 5
RUN_TYPES[QTABLE_GS5][Q_TABLE_INPUT] = "meleeAgent_qGS5_qtable"
RUN_TYPES[QTABLE_GS5][PARAMS] = QTableParamsExplorationDecay(NumStateVal(5), NumActions(5), propogateReward = False)
RUN_TYPES[QTABLE_GS5][NN] = ""
RUN_TYPES[QTABLE_GS5][Q_TABLE] = "battleMngr_qGS5_qtable"
RUN_TYPES[QTABLE_GS5][HISTORY] = "battleMngr_qGS5_replayHistory"
RUN_TYPES[QTABLE_GS5][RESULTS] = "battleMngr_qGS5_result"

RUN_TYPES[QTABLE_GS5_DIRECT] = {}
RUN_TYPES[QTABLE_GS5_DIRECT][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS5_DIRECT][GRIDSIZE_key] = 5
RUN_TYPES[QTABLE_GS5_DIRECT][PARAMS] = QTableParamsExplorationDecay(NumStateVal(5), NumActions(5))
RUN_TYPES[QTABLE_GS5_DIRECT][NN] = ""
RUN_TYPES[QTABLE_GS5_DIRECT][Q_TABLE] = "battleMngr_qGS5_DirectAttack_qtable"
RUN_TYPES[QTABLE_GS5_DIRECT][HISTORY] = "battleMngr_qGS5_DirectAttack_replayHistory"
RUN_TYPES[QTABLE_GS5_DIRECT][RESULTS] = "battleMngr_qGS5_DirectAttack_result"

RUN_TYPES[DQN_GS5_DIRECT] = {}
RUN_TYPES[DQN_GS5_DIRECT][TYPE] = "NN"
RUN_TYPES[DQN_GS5_DIRECT][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_DIRECT][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5))
RUN_TYPES[DQN_GS5_DIRECT][NN] = "battleMngr_dqnGS5_DirectAttack_DQN"
RUN_TYPES[DQN_GS5_DIRECT][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_DIRECT][HISTORY] = "battleMngr_dqnGS5_DirectAttack_replayHistory"
RUN_TYPES[DQN_GS5_DIRECT][RESULTS] = "battleMngr_dqnGS5_DirectAttack_result"


RUN_TYPES[QTABLE_GS5_DIRECT_MULTI] = {}
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][GRIDSIZE_key] = 5
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][PARAMS] = QTableParamsExplorationDecay(NumStateVal(5), NumActions(5))
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][NN] = ""
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][Q_TABLE] = "battleMngr_qGS5_DirectMultiAttack_qtable"
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][HISTORY] = "battleMngr_qGS5_DirectMultiAttack_replayHistory"
RUN_TYPES[QTABLE_GS5_DIRECT_MULTI][RESULTS] = "battleMngr_qGS5_DirectMultiAttack_result"

RUN_TYPES[DQN_GS5_DIRECT_MULTI] = {}
RUN_TYPES[DQN_GS5_DIRECT_MULTI][TYPE] = "NN"
RUN_TYPES[DQN_GS5_DIRECT_MULTI][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_DIRECT_MULTI][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5))
RUN_TYPES[DQN_GS5_DIRECT_MULTI][NN] = "battleMngr_dqnGS5_DirectMultiAttack_DQN"
RUN_TYPES[DQN_GS5_DIRECT_MULTI][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_DIRECT_MULTI][HISTORY] = "battleMngr_dqnGS5_DirectMultiAttack_replayHistory"
RUN_TYPES[DQN_GS5_DIRECT_MULTI][RESULTS] = "battleMngr_dqnGS5_DirectMultiAttack_result"


RUN_TYPES[DQN_GS5_DIRECT_BASE] = {}
RUN_TYPES[DQN_GS5_DIRECT_BASE][TYPE] = "NN"
RUN_TYPES[DQN_GS5_DIRECT_BASE][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_DIRECT_BASE][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), state2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_DIRECT_BASE][NN] = "battleMngr_dqnGS5_DirectBaseAttack_DQN"
RUN_TYPES[DQN_GS5_DIRECT_BASE][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_DIRECT_BASE][HISTORY] = "battleMngr_dqnGS5_DirectBaseAttack_replayHistory"
RUN_TYPES[DQN_GS5_DIRECT_BASE][RESULTS] = "battleMngr_dqnGS5_DirectBaseAttack_result"

RUN_TYPES[DQN_GS5_ARMY_ATTACK] = {}
RUN_TYPES[DQN_GS5_ARMY_ATTACK][TYPE] = "NN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_ARMY_ATTACK][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), state2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_ARMY_ATTACK][NN] = "battleMngr_dqnGS5_ArmyAttack_DQN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_ARMY_ATTACK][HISTORY] = "battleMngr_dqnGS5_ArmyAttack_replayHistory"
RUN_TYPES[DQN_GS5_ARMY_ATTACK][RESULTS] = "battleMngr_dqnGS5_ArmyAttack_result"

RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL] = {}
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][TYPE] = "NN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), state2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][NN] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal_DQN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][HISTORY] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal_replayHistory"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL][RESULTS] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal_result"

RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT] = {}
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][TYPE] = "NN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), state2Monitor = BaseTestSet(5), nn_Func=build_dqn_0init)
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][NN] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal0Init_DQN"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][HISTORY] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal0Init_replayHistory"
RUN_TYPES[DQN_GS5_ARMY_ATTACK_NOILLIGAL_0INIT][RESULTS] = "battleMngr_dqnGS5_ArmyAttack_NoIlligal0Init_result"

RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL] = {}
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][TYPE] = "NN"
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), state2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][NN] = "battleMngr_dqnGS5_BaseAttack_NoIlligal_DQN"
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][HISTORY] = "battleMngr_dqnGS5_BaseAttack_NoIlligal_replayHistory"
RUN_TYPES[DQN_GS5_BASE_ATTACK_NOILLIGAL][RESULTS] = "battleMngr_dqnGS5_BaseAttack_NoIlligal_result"



RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS] = {}
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][TYPE] = "NN"
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][BUILDING_VALUES_key] = True
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), state2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][NN] = "battleMngr_dqnGS5_DirectBaseAttack_BuildingWeights_DQN"
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][HISTORY] = "battleMngr_dqnGS5_DirectBaseAttack_BuildingWeights_replayHistory"
RUN_TYPES[DQN_GS5_DIRECT_BASE_BUILDING_WEIGHTS][RESULTS] = "battleMngr_dqnGS5_DirectBaseAttack_BuildingWeights_result"


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

        runTypeArg = ATTACK_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("\n\nplay type not entered correctly\n\n")
            exit() 

        runType = RUN_TYPES[runTypeArg.pop()]

        # states and action:
        self.gridSize = runType[GRIDSIZE_key]
        self.numActions = self.gridSize * self.gridSize + 1
        self.current_action = None

        # insert state params
        self.state_startSelfMat = 0
        self.state_startEnemyMat = self.gridSize * self.gridSize
        self.state_timeLineIdx = 2 * self.gridSize * self.gridSize

        self.state_size = 2 * self.gridSize * self.gridSize + 1
        self.inputStateSize = 2 * self.gridSize * self.gridSize + 1

        self.qTableInput = QLearningTable(QTableParams(self.inputStateSize, NUM_PHYSICAL_ACTIONS), runType[Q_TABLE_INPUT])

        self.terminalStates = self.TerminalStates()
        
        self.illigalMoveReward = -1.0

        self.isNaive = False
        # create decision maker
        if runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            self.decisionMaker = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            self.decisionMaker = UserPlay()
        elif runType[TYPE] == 'Naive':
            self.isNaive = True
            self.decisionMaker = NaiveDecisionMaker(runType[RESULTS])
        else:
            print("\n\ninvalid run type\n\n")
            exit()



    def step(self, obs):
        super(Attack, self).step(obs)
        if obs.first():
            return self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
            
        self.sumReward += obs.reward
        self.CreateState(obs)
        

        if self.errorOccur:
            return DO_NOTHING_SC2_ACTION

        self.numStep += 1
        sc2Action = DO_NOTHING_SC2_ACTION
        self.Learn()
        self.current_physical_action = self.ChooseAction()
        time.sleep(STEP_DURATION)

        if self.current_physical_action > ACTION_DO_NOTHING and self.selfLocCoord != None:   
            goTo = []
            for i in range(2):
                loc = self.selfLocCoord[i] + GOTO_CHANGE[self.current_physical_action][i]
                loc = min(max(loc,0), SC2_Params.SCREEN_SIZE - 1)
                goTo.append(loc)

            if SC2_Actions.MOVE_IN_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.MOVE_IN_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
                
        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0

        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.current_action = None
        self.current_physical_action = None
        
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
    
    def ChooseAction(self):
        if self.isNaive:
            qValues = self.QValues()
            self.current_action = self.decisionMaker.choose_action(qValues)
        else:
            self.current_action = self.decisionMaker.choose_action(self.current_state)

        if self.current_action == self.gridSize * self.gridSize:
            return ACTION_DO_NOTHING
        elif self.current_action < self.gridSize * self.gridSize and self.current_state[self.state_startEnemyMat + self.current_action] > 0:
            inputState = np.zeros(self.inputStateSize, dtype = int)
            inputState[self.state_startSelfMat:self.state_startEnemyMat] = self.current_state[self.state_startSelfMat:self.state_startEnemyMat]
            inputState[self.state_startEnemyMat + self.current_action] = 1
            inputState[self.state_timeLineIdx] = self.current_state[self.state_timeLineIdx]
            
            a, _ = self.qTableInput.choose_absolute_action(str(inputState))
            return a
        else:
            #print("illigal reward!!\n")
            self.prevReward += self.illigalMoveReward
            return ACTION_DO_NOTHING

    def Learn(self):
        if self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, self.prevReward, self.current_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.prevReward = 0

    def CreateState(self, obs):
        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
    
        selfPower = self.GetSelfLocationAndPower(obs)

        selfIdx  = self.state_startSelfMat + self.GetScaledIdx(self.selfLocCoord)
        self.current_state[selfIdx] = selfPower

        self.GetEnemyLoc(obs)

        self.current_state[self.state_timeLineIdx] = int(self.numStep / TIME_LINE_BUCKETING)

    def GetSelfLocationAndPower(self, obs):
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]

        selfLocY, selfLocX = (screenMap == SC2_Params.PLAYER_SELF).nonzero()
        
        if len(selfLocX) > 0:
            xMid = int(sum(selfLocX) / len(selfLocX))
            yMid = int(sum(selfLocY) / len(selfLocY))
            self.selfLocCoord = [yMid, xMid]
        else:
            selfLocY, selfLocX = (obs.observation["screen"][SC2_Params.UNIT_DENSITY] > 1).nonzero()
            if len(selfLocX) > 0:
                xMid = int(sum(selfLocX) / len(selfLocX))
                yMid = int(sum(selfLocY) / len(selfLocY))
                self.selfLocCoord = [yMid, xMid]

        selfPower = math.ceil(len(selfLocY) /  NUM_UNIT_SCREEN_PIXELS)
        return selfPower

    def GetEnemyLoc(self, obs):
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        e_y, e_x = (screenMap == SC2_Params.PLAYER_HOSTILE).nonzero()
        enemyPoints, enemyPower = CenterPoints(e_y, e_x)

        
        for i in range(len(enemyPoints)):
            idx = self.state_startEnemyMat + self.GetScaledIdx(enemyPoints[i])
            power = math.ceil(enemyPower[i] /  NUM_UNIT_SCREEN_PIXELS)
            self.current_state[idx] = power

        

    def GetScaledIdx(self, screenCord):
        locX = screenCord[SC2_Params.X_IDX]
        locY = screenCord[SC2_Params.Y_IDX]

        yScaled = int((locY / SC2_Params.SCREEN_SIZE) * self.gridSize)
        xScaled = int((locX / SC2_Params.SCREEN_SIZE) * self.gridSize)

        return xScaled + yScaled * self.gridSize

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
    def QValues(self):
        valState = np.ones(self.gridSize * self.gridSize, dtype=float) * NON_VALID_LOCATION_QVAL

        inputState = np.zeros(self.inputStateSize, dtype = int)
        inputState[self.state_startSelfMat:self.state_startEnemyMat] = self.current_state[self.state_startSelfMat:self.state_startEnemyMat]
        for i in range (self.gridSize * self.gridSize):
            if self.current_state[self.state_startEnemyMat + i] > 0:
                inputState[self.state_startEnemyMat + i] = 1
                a, v = self.qTableInput.choose_absolute_action(str(inputState))
                inputState[self.state_startEnemyMat + i] = 0
                valState[i] = v

        return valState
    def PrintState(self):
        print("\n\nstate: timeline =", self.current_state[self.state_timeLineIdx])
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                idx = self.state_startSelfMat + x + y * self.gridSize
                print(int(self.current_state[idx]), end = '')
            print(end = '  |  ')
            
            for x in range(self.gridSize):
                idx = self.state_startEnemyMat + x + y * self.gridSize
                print(int(self.current_state[idx]), end = '')

            print('||')



class DirectAttack(base_agent.BaseAgent):
    def __init__(self):        
        super(DirectAttack, self).__init__()

        runTypeArg = DIRECT_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("\n\nplay type not entered correctly\n\n")
            exit() 

        runType = RUN_TYPES[runTypeArg.pop()]

        # insert state params
        self.gridSize = runType[GRIDSIZE_key]
        self.state_startSelfMat = 0
        self.state_startEnemyMat = self.gridSize * self.gridSize
        self.state_lastValidAttackActionIdx = 2 * self.gridSize * self.gridSize
        self.state_timeLineIdx = 2 * self.gridSize * self.gridSize + 1

        self.state_size = 2 * self.gridSize * self.gridSize + 2

        self.terminalStates = self.TerminalStates()
        
        self.illigalMoveReward = -1.0

        # create decision maker
        if runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            self.decisionMaker = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            self.decisionMaker = UserPlay(False)
        else:
            print("\n\ninvalid run type\n\n")
            exit()

        # states and action:
        self.current_action = None
        self.lastValidAttackAction = None
        self.enemyArmyGridLoc2ScreenLoc = {}

    def step(self, obs):
        super(DirectAttack, self).step(obs)
        if obs.first():
            return self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
            
        self.sumReward += obs.reward
        self.CreateState(obs)
        # print("\n")
        # self.PrintState()
        # print(self.enemyArmyGridLoc2ScreenLoc)
        if self.errorOccur:
            return DO_NOTHING_SC2_ACTION

        self.numStep += 1
        sc2Action = DO_NOTHING_SC2_ACTION
        self.Learn()
        self.current_action = self.decisionMaker.choose_action(self.current_state)
        time.sleep(STEP_DURATION)

        if self.current_action < self.gridSize * self.gridSize:
            if self.current_action in self.enemyArmyGridLoc2ScreenLoc.keys():   
                goTo = self.enemyArmyGridLoc2ScreenLoc[self.current_action].copy()
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

    def CreateState(self, obs):
        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyLoc(obs)
        
        self.current_state[self.state_lastValidAttackActionIdx] = self.lastValidAttackAction
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

        # screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]

        # selfLocY, selfLocX = (screenMap == SC2_Params.PLAYER_SELF).nonzero()
        
        # if len(selfLocX) > 0:
        #     xMid = int(sum(selfLocX) / len(selfLocX))
        #     yMid = int(sum(selfLocY) / len(selfLocY))
        #     self.selfLocCoord = [yMid, xMid]
        # else:
        #     selfLocY, selfLocX = (obs.observation["screen"][SC2_Params.UNIT_DENSITY] > 1).nonzero()
        #     if len(selfLocX) > 0:
        #         xMid = int(sum(selfLocX) / len(selfLocX))
        #         yMid = int(sum(selfLocY) / len(selfLocY))
        #         self.selfLocCoord = [yMid, xMid]

        # selfPower = math.ceil(len(selfLocY) /  NUM_UNIT_SCREEN_PIXELS)
        # return selfPower

    def GetEnemyLoc(self, obs):
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        e_y, e_x = (screenMap == SC2_Params.PLAYER_HOSTILE).nonzero()
        enemyPoints, enemyPower = CenterPoints(e_y, e_x)

        self.enemyArmyGridLoc2ScreenLoc = {}
        for i in range(len(enemyPoints)):
            idx = self.GetScaledIdx(enemyPoints[i])
            power = math.ceil(enemyPower[i] /  NUM_UNIT_SCREEN_PIXELS)
            if idx in self.enemyArmyGridLoc2ScreenLoc.keys():
                self.current_state[self.state_startEnemyMat + idx] += power
                self.enemyArmyGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyArmyGridLoc2ScreenLoc[idx], enemyPoints[i])
            else:
                self.current_state[self.state_startEnemyMat + idx] = power
                self.enemyArmyGridLoc2ScreenLoc[idx] = enemyPoints[i]

        
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

            print('||')



class BaseAttack(base_agent.BaseAgent):
    def __init__(self):        
        super(BaseAttack, self).__init__()

        runTypeArg = BASE_TYPES.intersection(sys.argv)
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
        self.state_lastValidAttackActionIdx = 3 * self.gridSize * self.gridSize
        self.state_timeLineIdx = 3 * self.gridSize * self.gridSize + 1

        self.state_size = 3 * self.gridSize * self.gridSize + 2
        self.terminalStates = self.TerminalStates()
        
        if runArg.find("NoIlligal") >= 0:
            self.illigalMoveReward = 0
            self.illigalmoveSolveInModel = True
        else:
            self.illigalMoveReward = -1.0
            self.illigalmoveSolveInModel = False

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
            self.decisionMaker = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            for spec in TerranUnit.BUILDING_SPEC.values():
                self.BuildingValues[spec.name] = spec.destroyRewards
            self.decisionMaker = UserPlay(False)
        else:
            print("\n\ninvalid run type\n\n")
            exit()


        
        self.lastValidAttackAction = None
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.enemyBuildingGridLoc2ScreenLoc = {}


    def step(self, obs):
        super(BaseAttack, self).step(obs)
        if obs.first():
            return self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
            
        self.sumReward += obs.reward
        self.CreateState(obs)

        # print("\n")
        # self.PrintState()
        # print(self.enemyArmyGridLoc2ScreenLoc)
        # print(self.enemyBuildingGridLoc2ScreenLoc)
        if self.errorOccur:
            return DO_NOTHING_SC2_ACTION

        self.numStep += 1
        sc2Action = DO_NOTHING_SC2_ACTION
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
            if np.random.uniform() > self.decisionMaker.ExploreProb():
                valVec = self.decisionMaker.actionValuesVec(self.current_state)   
                validActions = self.ValidActions()
                random.shuffle(validActions)
                validVal = valVec[validActions]
                action = validActions[validVal.argmax()]
            else:
                action = np.random.choice(self.numActions) 
        else:
            action = self.decisionMaker.choose_action(self.current_state)
        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyArmyLoc(obs)
        self.GetEnemyBuildingLoc(obs)
        
        self.current_state[self.state_lastValidAttackActionIdx] = self.lastValidAttackAction
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

