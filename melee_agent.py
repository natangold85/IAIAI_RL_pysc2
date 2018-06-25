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

from utils_tables import DQN_PARAMS
from utils_tables import QTableParams


from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp

NUM_ENEMIES = 2

STATE_GRID_SIZE = 10
SELF_MAT_START = 0
ENEMY_MAT_START = STATE_GRID_SIZE * STATE_GRID_SIZE
TIME_LINE_BUCKETING = 25
STATE_TIME_LINE_IDX = 2 * STATE_GRID_SIZE * STATE_GRID_SIZE
STATE_SIZE = 2 * STATE_GRID_SIZE * STATE_GRID_SIZE + 1

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

for key,value in TerranUnit.UNIT_SPEC.items():
    if value.name == "marine":
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels


# possible types of play
QTABLE_SIMPLE = "QTable"
DQN_NN_SIGMOID = "dqn_Sig"
DQN_NN_SIGMOID_MIDDLE_REWARDS = "dqn_SigRewards"

QTABLE_5ACTIONS = "QTable_5Actions"
DQN_NN_SIGMOID_5ACTIONS = "dqn_Sig_5Actions"
DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS = "dqn_SigRewards_5Actions"

REWARD_PROPAGATION = 'Q_rewardPropogation'

TEST = "test"

USER_PLAY = 'play'

ALL_TYPES = set([TEST, USER_PLAY,
            QTABLE_SIMPLE, DQN_NN_SIGMOID, 
            DQN_NN_SIGMOID_5ACTIONS, QTABLE_5ACTIONS, DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS,
            DQN_NN_SIGMOID_MIDDLE_REWARDS])

# table type
TYPE = "type"
NN = "nn"
Q_TABLE = "q"
T_TABLE = "t"
HISTORY = "hist"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'
REWARDS = "rewards"
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
        output = tf.nn.sigmoid(tf.contrib.layers.fully_connected(fc1, numActions), name=nameChar+'o') * 2 - 1
    return output

def build_nn_sig_processedState(x, numActions, scope = 'dqn', nameChar = 'l'):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512)
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid) * 2 - 1

        #output = tf.nn.sigmoid(tf.contrib.layers.fully_connected(fc1, numActions)) * 2 - 1
    return output

def build_nn_processedState(x, numActions, scope = 'dqn', nameChar = 'l'):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512)
        output = tf.contrib.layers.fully_connected(fc1, numActions)
    return output

def build_nn_4double(x, numActions, w_init, b_init, scope = 'dqn', nameChar = 'l'):
    with tf.variable_scope(scope):
        # Fully connected layers
        l1 = tf.layers.dense(x, 512, tf.nn.relu, kernel_initializer=w_init, bias_initializer=b_init, name=nameChar+'0')
        output = tf.nn.sigmoid(tf.layers.dense(l1, numActions, kernel_initializer=w_init, bias_initializer=b_init, name=nameChar+'1')) * 2 - 1
    return output

# RUN_TYPES[TEST] = {}
# RUN_TYPES[TEST][TYPE] = "NN"
# RUN_TYPES[TEST][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, 1, build_nn_sig_processedState, True)
# RUN_TYPES[TEST][NN] = "test_DQN"
# RUN_TYPES[TEST][REWARDS] = [-0.1, 0.1]
# RUN_TYPES[TEST][Q_TABLE] = ""
# RUN_TYPES[TEST][HISTORY] = ""
# RUN_TYPES[TEST][R_TABLE] = ""
# RUN_TYPES[TEST][RESULTS] = ""


RUN_TYPES[DQN_NN_SIGMOID] = {}
RUN_TYPES[DQN_NN_SIGMOID][TYPE] = "NN"
RUN_TYPES[DQN_NN_SIGMOID][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, 1, build_nn_sig_processedState, True)
RUN_TYPES[DQN_NN_SIGMOID][NN] = "meleeAgent1_Sig_DQN"
RUN_TYPES[DQN_NN_SIGMOID][Q_TABLE] = "meleeAgent1_Sig_qtable"
RUN_TYPES[DQN_NN_SIGMOID][HISTORY] = "meleeAgent1_Sig_historyReplay"
RUN_TYPES[DQN_NN_SIGMOID][R_TABLE] = ""
RUN_TYPES[DQN_NN_SIGMOID][RESULTS] = "meleeAgent1_Sig_result"

RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS] = {}
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][TYPE] = "NN"
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][PARAMS] = DQN_PARAMS(STATE_SIZE, 5, 1, build_nn_sig_processedState, True)
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][NN] = "meleeAgent1_Sig5Actions_DQN"
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][Q_TABLE] = "meleeAgent1_Sig5Actions_qtable"
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][HISTORY] = "meleeAgent1_Sig5Actions_historyReplay"
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][R_TABLE] = ""
RUN_TYPES[DQN_NN_SIGMOID_5ACTIONS][RESULTS] = "meleeAgent1_Sig5Actions_result"

RUN_TYPES[QTABLE_5ACTIONS] = {}
RUN_TYPES[QTABLE_5ACTIONS][TYPE] = "QReplay"
RUN_TYPES[QTABLE_5ACTIONS][PARAMS] = QTableParams(STATE_SIZE, 5)
RUN_TYPES[QTABLE_5ACTIONS][NN] = ""
RUN_TYPES[QTABLE_5ACTIONS][Q_TABLE] = "meleeAgent1_Q5Actions_qtable"
RUN_TYPES[QTABLE_5ACTIONS][HISTORY] = "meleeAgent1_Q5Actions_history"
RUN_TYPES[QTABLE_5ACTIONS][RESULTS] = "meleeAgent1_Q5Actions_result"


RUN_TYPES[QTABLE_SIMPLE] = {}
RUN_TYPES[QTABLE_SIMPLE][TYPE] = "QReplay"
RUN_TYPES[QTABLE_SIMPLE][PARAMS] = QTableParams(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE_SIMPLE][NN] = ""
RUN_TYPES[QTABLE_SIMPLE][Q_TABLE] = "meleeAgent1_Q_qtable"
RUN_TYPES[QTABLE_SIMPLE][HISTORY] = "meleeAgent1_Q_history"
RUN_TYPES[QTABLE_SIMPLE][RESULTS] = "meleeAgent1_Q_result"

RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS] = {}
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][TYPE] = "NN"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][PARAMS] = DQN_PARAMS(STATE_SIZE, 5, 1, build_nn_sig_processedState, True)
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][REWARDS] = [-0.1, 0.1, 0]
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][NN] = "meleeAgent1_Sig5Actions_MiddleRewards_DQN"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][Q_TABLE] = "meleeAgent1_Sig5Actions_MiddleRewards_qtable"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][HISTORY] = "meleeAgent1_Sig5Actions_MiddleRewards_historyReplay"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][R_TABLE] = ""
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS_5ACTIONS][RESULTS] = "meleeAgent1_Sig5Actions_MiddleRewards_result"


RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS] = {}
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][TYPE] = "NN"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, 1, build_nn_sig_processedState, True)
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][REWARDS] = [-0.1, 0.1, 0]
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][NN] = "meleeAgent1_Sig_MiddleRewards_DQN"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][Q_TABLE] = "meleeAgent1_Sig_MiddleRewards_qtable"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][HISTORY] = "meleeAgent1_Sig_MiddleRewards_historyReplay"
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][R_TABLE] = ""
RUN_TYPES[DQN_NN_SIGMOID_MIDDLE_REWARDS][RESULTS] = "meleeAgent1_Sig_MiddleRewards_result"

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
        self.terminalStates = self.TerminalStates()

        runTypeArg = ALL_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("play type not entered correctly")
            exit(1) 

        runType = RUN_TYPES[runTypeArg.pop()]
        if runType[TYPE] == 'all':
            params = runType[PARAMS]
            self.tables = TableMngr(NUM_ACTIONS, STATE_SIZE, runType[Q_TABLE], runType[RESULTS], params[0]) 
        
        elif runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            self.tables = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])   
            if REWARDS in runType.keys():
                self.rewardIlligalMove = runType[REWARDS][0]
                self.rewardKill = runType[REWARDS][1]
                self.rewardDeath = runType[REWARDS][2]
            else:
                self.rewardIlligalMove = 0
                self.rewardKill = 0
                self.rewardDeath = 0

        elif runType[TYPE] == 'play':
            self.rewardIlligalMove = -0.1
            self.rewardKill = 0.1
            self.tables = UserPlay(False)

        # states and action:
        self.current_action = None

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
           
        self.selfLocCoord = []
        
        self.numEpisode = 0

        self.prevReward = 0
        self.prevScore = 0

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
        self.AddKillReward(obs)
        
        self.Learn(self.prevReward)
        self.current_action = self.tables.choose_action(self.current_state)

        if self.numEpisode % 100 == 99:
            print("\n\nstate: time=", self.current_state[STATE_TIME_LINE_IDX], "action=" , self.current_action, "vals =", self.tables.actionValuesVec(self.current_state))
            for i in range(STATE_GRID_SIZE * STATE_GRID_SIZE):
                print(self.current_state[i], end = ' ')
            print("")
            for i in range(STATE_GRID_SIZE * STATE_GRID_SIZE):
                print(self.current_state[100 + i], end = ' ')
            print("")

        time.sleep(STEP_DURATION)

        if self.current_action > ACTION_DO_NOTHING and self.selfLocCoord != None:   
            goTo = []
            for i in range(2):
                loc = self.selfLocCoord[i] + GOTO_CHANGE[self.current_action][i]
                loc = min(max(loc,0), SC2_Params.SCREEN_SIZE - 1)
                goTo.append(loc)
            
            self_y, self_x = (obs.observation["screen"][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
            
            if self.MoveOutOfBounds(goTo):
                self.prevReward += self.rewardIlligalMove
                

            if SC2_Actions.MOVE_IN_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.MOVE_IN_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
                
        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.selfLocCoord = None
        self.enemyCoord = None
                
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
            self.tables.learn(self.current_state.copy(), self.current_action, float(reward), s_, True)

        score = obs.observation["score_cumulative"][0]
        self.tables.end_run(reward, score)

        self.numEpisode += 1

    def MoveOutOfBounds(self, coord):
        for i in range(len(coord)):
            if coord[i] < SCREEN_MIN[i] or coord[i] > SCREEN_MAX[i]:
                return True
        
        return False

    def Learn(self, r):
        if self.current_action is not None:
            self.tables.learn(self.previous_state.copy(), self.current_action, r, self.current_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.prevReward = 0

    def CreateState(self, obs):
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        
        self.GetSelfLoc(screenMap)     
        self.GetEnemyLoc(screenMap)

        self.current_state[STATE_TIME_LINE_IDX] = int(self.numStep / TIME_LINE_BUCKETING)

    def GetSelfLoc(self, screenMap):
        self_y, self_x = (screenMap == SC2_Params.PLAYER_SELF).nonzero()
        if len(self_x) > 0:
            xMid = int(sum(self_x) / len(self_x))
            yMid = int(sum(self_y) / len(self_y))
            
            scaled_y = int(yMid * STATE_GRID_SIZE / SC2_Params.SCREEN_SIZE)
            scaled_x = int(xMid * STATE_GRID_SIZE / SC2_Params.SCREEN_SIZE)        
            
            idx = scaled_x + scaled_y * STATE_GRID_SIZE
            self.current_state[SELF_MAT_START + idx] = 1
            self.selfLocCoord = [yMid, xMid]

    def GetEnemyLoc(self, screenMap):
        e_y, e_x = (screenMap == SC2_Params.PLAYER_HOSTILE).nonzero()
        enemyLocations = {}
        for i in range(len(e_y)):
            scaled_y = int(e_y[i] * STATE_GRID_SIZE / SC2_Params.SCREEN_SIZE)
            scaled_x = int(e_x[i] * STATE_GRID_SIZE / SC2_Params.SCREEN_SIZE)
            
            idx = scaled_x + scaled_y * STATE_GRID_SIZE
            if idx not in enemyLocations.keys():
                enemyLocations[idx] = 1
            else:
                enemyLocations[idx] += 1
        

        nonComplete = 0
        for key,val in enemyLocations.items():
            if val >= NUM_UNIT_SCREEN_PIXELS:
                c = int(val / NUM_UNIT_SCREEN_PIXELS)
                self.current_state[ENEMY_MAT_START + key] = c
                val -= c * NUM_UNIT_SCREEN_PIXELS
            if val > 0:
                nonComplete += val
                if nonComplete > int(0.5 * NUM_UNIT_SCREEN_PIXELS):
                    self.current_state[ENEMY_MAT_START + key] += 1
                    nonComplete -= NUM_UNIT_SCREEN_PIXELS

            enemyLocations[key] = 0         
    
    def TerminalStates(self):
        tStates = {}
        state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        state[0] = -1
        tStates["win"] = state.copy()
        state[0] = -2
        tStates["tie"] = state.copy()
        state[0] = -3
        tStates["loss"] = state.copy()

        return tStates
        
    def AddKillReward(self , obs):
        cumulativeScore = obs.observation["score_cumulative"][0]
        if cumulativeScore > self.prevScore:
            self.prevReward += self.rewardKill
        
        self.prevScore = cumulativeScore
        

