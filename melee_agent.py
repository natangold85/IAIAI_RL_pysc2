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
from utils_tables import UserPlay
from utils_tables import QLearningTable

from utils_tables import DQN_PARAMS
from utils_tables import QTableParams
from utils_tables import QTableParamsExplorationDecay

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp
from utils import CenterPoints

DEFAULT_STATE_GRID_SIZE = 10
TIME_LINE_BUCKETING = 25

def NumStateVal(gridSize):
    return 2 * gridSize * gridSize + 1

ACTION_DO_NOTHING = 0
ACTION_NORTH = 1
ACTION_SOUTH = 2
ACTION_EAST = 3
ACTION_WEST = 4
ACTION_NORTH_EAST = 5
ACTION_NORTH_WEST = 6
ACTION_SOUTH_EAST = 7
ACTION_SOUTH_WEST = 8

NUM_ACTIONS =  9

NUM_UNIT_SCREEN_PIXELS = 0

SCREEN_MIN = [3,3]
SCREEN_MAX = [59,80]

for key,value in TerranUnit.ARMY_SPEC.items():
    if value.name == "marine":
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels

# possible types of play
DQN_NN_SIGMOID_GS10 = "dqnSigGS10"
DQN_NN_SIGMOID_GS5 = "dqnSigGS5"
QTABLE_GS10 = 'qGS10'
QTABLE_GS5 = 'qGS5'
QTABLE_GS5_TEST = 'qGS5Test'

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE_GS10, QTABLE_GS5, DQN_NN_SIGMOID_GS10, DQN_NN_SIGMOID_GS5, QTABLE_GS5_TEST])

# table type
TYPE = "type"
NN = "nn"
Q_TABLE = "q"
T_TABLE = "t"
HISTORY = "hist"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'
GS_key = 'gridsize'

# table names
RUN_TYPES = {}

def build_nn_sigmoid(x, numActions, scope = 'dqn', nameChar = 'l'):
    with tf.variable_scope(scope):
        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu, name=nameChar+'0' )
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, name=nameChar+'1')
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu, name=nameChar+'2')

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, name=nameChar+'3')
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid, name=nameChar+'o') * 2 - 1
    return output

def build_nn_sig_processedState(x, numActions, scope = 'dqn', nameChar = 'l'):
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

def States2Check(gridSize):
    eOffset = gridSize * gridSize
    tIdx = 2* gridSize * gridSize
    ret = []
    s = np.zeros(1 + 2* gridSize * gridSize, dtype = int)
    
    s[1] = 2
    s[eOffset + 15] = 1
    s[tIdx] = 1
    ret.append(s)

    s1 = np.zeros(1 + 2* gridSize * gridSize, dtype = int)
    s1[1] = 2
    s1[eOffset + 15] = 1
    s1[tIdx] = 9
    ret.append(s1)

    s2 = np.zeros(1 + 2* gridSize * gridSize, dtype = int)
    
    s2[14] = 2
    s2[eOffset + 15] = 1
    ret.append(s2)

    return ret


RUN_TYPES[DQN_NN_SIGMOID_GS10] = {}
RUN_TYPES[DQN_NN_SIGMOID_GS10][TYPE] = "NN"
RUN_TYPES[DQN_NN_SIGMOID_GS10][PARAMS] = DQN_PARAMS(NumStateVal(10), NUM_ACTIONS, nn_Func = build_nn_sig_processedState)
RUN_TYPES[DQN_NN_SIGMOID_GS10][NN] = "meleeAgent_Sig_GS10_DQN"
RUN_TYPES[DQN_NN_SIGMOID_GS10][Q_TABLE] = "meleeAgent_Sig_GS10_qtable"
RUN_TYPES[DQN_NN_SIGMOID_GS10][HISTORY] = "meleeAgent_Sig_GS10_historyReplay"
RUN_TYPES[DQN_NN_SIGMOID_GS10][RESULTS] = "meleeAgent_Sig_GS10_result"


RUN_TYPES[DQN_NN_SIGMOID_GS5] = {}
RUN_TYPES[DQN_NN_SIGMOID_GS5][TYPE] = "NN"
RUN_TYPES[DQN_NN_SIGMOID_GS5][PARAMS] = DQN_PARAMS(NumStateVal(5), NUM_ACTIONS, nn_Func = build_nn_sig_processedState, state2Monitor=States2Check(5))
RUN_TYPES[DQN_NN_SIGMOID_GS5][NN] = "meleeAgent_Sig_GS5_DQN"
RUN_TYPES[DQN_NN_SIGMOID_GS5][Q_TABLE] = "meleeAgent_Sig_GS5_qtable"
RUN_TYPES[DQN_NN_SIGMOID_GS5][HISTORY] = "meleeAgent_Sig_GS5_historyReplay"
RUN_TYPES[DQN_NN_SIGMOID_GS5][RESULTS] = "meleeAgent_Sig_GS5_result"
RUN_TYPES[DQN_NN_SIGMOID_GS5][GS_key] = 5

RUN_TYPES[QTABLE_GS10] = {}
RUN_TYPES[QTABLE_GS10][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS10][PARAMS] = QTableParamsExplorationDecay(NumStateVal(10), NUM_ACTIONS, propogateReward = False)
RUN_TYPES[QTABLE_GS10][NN] = ""
RUN_TYPES[QTABLE_GS10][Q_TABLE] = "meleeAgent_qGS10_qtable"
RUN_TYPES[QTABLE_GS10][HISTORY] = "meleeAgent_qGS10_replayHistory"
RUN_TYPES[QTABLE_GS10][RESULTS] = "meleeAgent_qGS10_result"

RUN_TYPES[QTABLE_GS5] = {}
RUN_TYPES[QTABLE_GS5][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS5][GS_key] = 5
RUN_TYPES[QTABLE_GS5][PARAMS] = QTableParamsExplorationDecay(NumStateVal(5), NUM_ACTIONS, propogateReward = False)
RUN_TYPES[QTABLE_GS5][NN] = ""
RUN_TYPES[QTABLE_GS5][Q_TABLE] = "meleeAgent_qGS5_qtable"
RUN_TYPES[QTABLE_GS5][HISTORY] = "meleeAgent_qGS5_replayHistory"
RUN_TYPES[QTABLE_GS5][RESULTS] = "meleeAgent_qGS5_result"

RUN_TYPES[QTABLE_GS5_TEST] = {}
RUN_TYPES[QTABLE_GS5_TEST][TYPE] = "QTest"
RUN_TYPES[QTABLE_GS5_TEST][GS_key] = 5
RUN_TYPES[QTABLE_GS5_TEST][PARAMS] = QTableParams(NumStateVal(5), NUM_ACTIONS, propogateReward = False, explorationProb=0.0)
RUN_TYPES[QTABLE_GS5_TEST][Q_TABLE] = "meleeAgent_qGS5_qtable"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"


DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

STEP_DURATION = 0

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
            print("play type not entered correctly")
            exit(1) 

        runType = RUN_TYPES[runTypeArg.pop()]

        # insert state params
        if GS_key in runType.keys():
            self.gridSize = runType[GS_key]
        else:
            self.gridSize = DEFAULT_STATE_GRID_SIZE

        self.state_startSelfMat = 0
        self.state_startEnemyMat = self.gridSize * self.gridSize
        self.state_timeLineIdx = 2 * self.gridSize * self.gridSize
        self.state_size = 2 * self.gridSize * self.gridSize + 1

        self.terminalStates = self.TerminalStates()

        # create learning mngr
        if runType[TYPE] == 'all':
            params = runType[PARAMS]
            self.tables = TableMngr(NUM_ACTIONS, self.state_size, runType[Q_TABLE], runType[RESULTS], params[0]) 
        elif runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            self.tables = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            self.tables = UserPlay()
        elif runType[TYPE] == 'QTest':
            self.tables = QLearningTable(runType[PARAMS], runType[Q_TABLE])


        # states and action:
        self.current_action = None

        self.current_state = np.zeros(self.state_size, dtype=np.int32, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int32, order='C')

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
        #self.current_action = self.tables.choose_action(self.current_state)
        self.current_action, _ = self.tables.choose_absolute_action(str(self.current_state))

        time.sleep(STEP_DURATION)

        if self.current_action > ACTION_DO_NOTHING and self.selfLocCoord != None:   
            goTo = []
            for i in range(2):
                loc = self.selfLocCoord[i] + GOTO_CHANGE[self.current_action][i]
                loc = min(max(loc,0), SC2_Params.SCREEN_SIZE - 1)
                goTo.append(loc)

            if SC2_Actions.MOVE_IN_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.MOVE_IN_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
                
        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0

        self.current_state = np.zeros(self.state_size, dtype=np.int32, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int32, order='C')

        self.selfLocCoord = None      
        self.enemyLocCoord = None  

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

        # if self.current_action is not None:
        #     self.tables.learn(self.current_state.copy(), self.current_action, float(reward), s_, True)

        # score = obs.observation["score_cumulative"][0]
        # self.tables.end_run(reward, score)

    def Learn(self):
        # if self.current_action is not None:
        #     self.tables.learn(self.previous_state.copy(), self.current_action, 0, self.current_state.copy())

        self.previous_state[:] = self.current_state[:]

    def CreateState(self, obs):
        self.current_state = np.zeros(self.state_size, dtype=np.int32, order='C')
    
        selfPower, enemyPower = self.GetLocationsAndPower(obs)

        selfIdx  = self.state_startSelfMat + self.GetScaledIdx(self.selfLocCoord)
        self.current_state[selfIdx] = selfPower

        enemyIdx = self.state_startEnemyMat + self.GetScaledIdx(self.enemyLocCoord)
        self.current_state[enemyIdx] = enemyPower

        self.current_state[self.state_timeLineIdx] = int(self.numStep / TIME_LINE_BUCKETING)

    def GetLocationsAndPower(self, obs):
        screenMap = obs.observation["feature_screen"][SC2_Params.PLAYER_RELATIVE]

        selfLocY, selfLocX = (screenMap == SC2_Params.PLAYER_SELF).nonzero()
        enemyLocY , enemyLocX = (screenMap == SC2_Params.PLAYER_HOSTILE).nonzero()
        
        if len(selfLocX) > 0:
            xMid = int(sum(selfLocX) / len(selfLocX))
            yMid = int(sum(selfLocY) / len(selfLocY))
            self.selfLocCoord = [yMid, xMid]
        else:
            selfLocY, selfLocX = (obs.observation["feature_screen"][SC2_Params.UNIT_DENSITY] > 1).nonzero()
            if len(selfLocX) > 0:
                xMid = int(sum(selfLocX) / len(selfLocX))
                yMid = int(sum(selfLocY) / len(selfLocY))
                self.selfLocCoord = [yMid, xMid]

        if len(enemyLocY) > 0:
            xMid = int(sum(enemyLocX) / len(enemyLocX))
            yMid = int(sum(enemyLocY) / len(enemyLocY))
            self.enemyLocCoord = [yMid, xMid]

        selfPower = math.ceil(len(selfLocY) /  NUM_UNIT_SCREEN_PIXELS)
        enemyPower = math.ceil(len(enemyLocY) /  NUM_UNIT_SCREEN_PIXELS)
        return selfPower, enemyPower

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
        
    def PrintState(self):
        print("\n\nstate: timeline =", self.current_state[self.state_timeLineIdx])
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                idx = self.state_startSelfMat + x + y * self.gridSize
                print(self.current_state[idx], end = '')
            print(end = '  |  ')
            
            for x in range(self.gridSize):
                idx = self.state_startEnemyMat + x + y * self.gridSize
                print(self.current_state[idx], end = '')
            print('||')

        

