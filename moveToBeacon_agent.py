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


STATE_GRID_SIZE = SC2_Params.SCREEN_SIZE

PROCESSED_STATE_GRID_SIZE = 10
SELF_MAT_START = 0
BEACON_MAT_START = PROCESSED_STATE_GRID_SIZE * PROCESSED_STATE_GRID_SIZE
PROCESSED_STATE_SIZE = 2 * PROCESSED_STATE_GRID_SIZE * PROCESSED_STATE_GRID_SIZE

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

# state details
STATE_SIZE = STATE_GRID_SIZE * STATE_GRID_SIZE


# possible types of play
SMART_EXPLORATION_GRID_SIZE_2 = 'smartExploration'
NAIVE_EXPLORATION_GRID_SIZE_2 = 'naiveExploration'
DQN_CMP_QTABLE = "dqnCmpQ"
DQN_CMP_QTABLE_FULL = "dqnCmpQFull_ProcessedState"
DQN_CMP_NEURAL_NETWORK = "dqnCmpNN"
DQN_CMP_NEURAL_NETWORK_FULL = "dqnCmpNNFull"
DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID = "dqnCmpNNFullSig"
DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED = "dqnCmpNNFullSig_ProcessedState"
DOUBLE_DQN_5UPDATE = "doubleDqn5Update"
DOUBLE_DQN_1UPDATE = "doubleDqn1Update"
DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED = "dqnCmpNNFull_ProcessedState"
ONLINE_HALLUCINATION = 'onlineHallucination'
REWARD_PROPAGATION = 'rewardPropogation'

USER_PLAY = 'play'

ALL_TYPES = set([SMART_EXPLORATION_GRID_SIZE_2, NAIVE_EXPLORATION_GRID_SIZE_2, 
            ONLINE_HALLUCINATION, REWARD_PROPAGATION, USER_PLAY,
            DQN_CMP_QTABLE, DQN_CMP_NEURAL_NETWORK, DQN_CMP_QTABLE_FULL, DQN_CMP_NEURAL_NETWORK_FULL, 
            DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID, 
            DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED, DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED,
            DOUBLE_DQN_5UPDATE, DOUBLE_DQN_1UPDATE])

# table type
TYPE = "type"
NN = "nn"
Q_TABLE = "q"
T_TABLE = "t"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'

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
        output = tf.nn.sigmoid(tf.contrib.layers.fully_connected(fc1, numActions)) * 2 - 1
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

RUN_TYPES[DOUBLE_DQN_5UPDATE] = {}
RUN_TYPES[DOUBLE_DQN_5UPDATE][TYPE] = "NN"
RUN_TYPES[DOUBLE_DQN_5UPDATE][PARAMS] = DQN_PARAMS(stateSize=PROCESSED_STATE_SIZE, numActions=NUM_ACTIONS, nn_Func=build_nn_4double, 
                                                                                outputGraph=True, isDoubleDQN=True, copyEvalToTarget = 5)
RUN_TYPES[DOUBLE_DQN_5UPDATE][NN] = "moveToBeacon_5Update_DoubleDQN_nn"
RUN_TYPES[DOUBLE_DQN_5UPDATE][Q_TABLE] = "moveToBeacon_5Update_DoubleDQN_qtable"
RUN_TYPES[DOUBLE_DQN_5UPDATE][T_TABLE] = ""
RUN_TYPES[DOUBLE_DQN_5UPDATE][R_TABLE] = ""
RUN_TYPES[DOUBLE_DQN_5UPDATE][RESULTS] = "moveToBeacon_5Update_DoubleDQN_results"

RUN_TYPES[DOUBLE_DQN_1UPDATE] = {}
RUN_TYPES[DOUBLE_DQN_1UPDATE][TYPE] = "NN"
RUN_TYPES[DOUBLE_DQN_1UPDATE][PARAMS] = DQN_PARAMS(stateSize=PROCESSED_STATE_SIZE, numActions=NUM_ACTIONS, nn_Func=build_nn_4double, 
                                                                                outputGraph=False, isDoubleDQN=True, copyEvalToTarget = 1)
RUN_TYPES[DOUBLE_DQN_1UPDATE][NN] = "moveToBeacon_1Update_DoubleDQN"
RUN_TYPES[DOUBLE_DQN_1UPDATE][Q_TABLE] = "moveToBeacon_1Update_DoubleDQN"
RUN_TYPES[DOUBLE_DQN_1UPDATE][T_TABLE] = ""
RUN_TYPES[DOUBLE_DQN_1UPDATE][R_TABLE] = ""
RUN_TYPES[DOUBLE_DQN_1UPDATE][RESULTS] = "moveToBeacon_1Update_DoubleDQN"


RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][PARAMS] = DQN_PARAMS(PROCESSED_STATE_SIZE, NUM_ACTIONS, 1, build_nn_sig_processedState, True, True)
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][NN] = "moveToBeacon_DQN_FullSig_ProcessedState"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][Q_TABLE] = "moveToBeacon_qtable_dqnCmpNN_FullSig_ProcessedState"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][T_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][R_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID_PROCESSED][RESULTS] = "moveToBeacon_result_dqnCmpNN_FullSig_ProcessedState"


RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][PARAMS] = DQN_PARAMS(PROCESSED_STATE_SIZE, NUM_ACTIONS, 1, build_nn_processedState, True, True)
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][NN] = "moveToBeacon_DQN_Full_ProcessedState"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][Q_TABLE] = "moveToBeacon_qtable_dqnCmpNN_Full_ProcessedState"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][T_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][R_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_PROCESSED][RESULTS] = "moveToBeacon_result_dqnCmpNN_Full_ProcessedState"


RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][PARAMS] = DQN_PARAMS(STATE_GRID_SIZE, NUM_ACTIONS, 1, build_nn_sigmoid, False, False)
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][NN] = "moveToBeacon_DQN_FullSig"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][Q_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][T_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][R_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL_SIGMOID][RESULTS] = "moveToBeacon_result_dqnCmpNN_FullSig"


RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][PARAMS] = DQN_PARAMS(STATE_GRID_SIZE, NUM_ACTIONS, 1, None, False, False)
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][NN] = "moveToBeacon_DQN_Full"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][Q_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][T_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][R_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][RESULTS] = "moveToBeacon_result_dqnCmpNN_Full"

RUN_TYPES[DQN_CMP_QTABLE] = {}
RUN_TYPES[DQN_CMP_QTABLE][TYPE] = "QReplay"
RUN_TYPES[DQN_CMP_QTABLE][PARAMS] = QTableParams(STATE_GRID_SIZE, NUM_ACTIONS)
RUN_TYPES[DQN_CMP_QTABLE][NN] = ""
RUN_TYPES[DQN_CMP_QTABLE][Q_TABLE] = "moveToBeacon_qtable_dqnCmpQ"
RUN_TYPES[DQN_CMP_QTABLE][T_TABLE] = ""
RUN_TYPES[DQN_CMP_QTABLE][RESULTS] = "moveToBeacon_result_dqnCmpQ"

RUN_TYPES[DQN_CMP_QTABLE_FULL] = {}
RUN_TYPES[DQN_CMP_QTABLE_FULL][TYPE] = "QReplay"
RUN_TYPES[DQN_CMP_QTABLE_FULL][PARAMS] = QTableParams(STATE_GRID_SIZE, NUM_ACTIONS)
RUN_TYPES[DQN_CMP_QTABLE_FULL][NN] = ""
RUN_TYPES[DQN_CMP_QTABLE_FULL][Q_TABLE] = "moveToBeacon_qtable_dqnCmpQ_Full"
RUN_TYPES[DQN_CMP_QTABLE_FULL][T_TABLE] = "moveToBeacon_history_dqnCmpQ_Full"
RUN_TYPES[DQN_CMP_QTABLE_FULL][RESULTS] = "moveToBeacon_result_dqnCmpQ_Full"


RUN_TYPES[DQN_CMP_NEURAL_NETWORK] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][PARAMS] = DQN_PARAMS(STATE_GRID_SIZE, NUM_ACTIONS, 0.25, None, False, False)
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][NN] = "moveToBeacon__DQN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][Q_TABLE] = "moveToBeacon_qtable_dqnCmpNN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][T_TABLE] = "moveToBeacon_history_dqnCmpNN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][RESULTS] = "moveToBeacon_result_dqnCmpNN"


RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"


SELF_VAL = 1
BEACON_VAL = 2

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

STEP_DURATION = 0
MARINE_SCREEN_SIZE_ONE_AXIS = 3

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

SCREEN_MIN = [4,4]
SCREEN_MAX = [58,79]
REWARD_ILLIGAL_MOVE = - 0.1

class Agent(base_agent.BaseAgent):
    def __init__(self):        
        super(Agent, self).__init__()

        runTypeArg = ALL_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("play type not entered correctly")
            exit(1) 

        self.terminalStates = self.TerminalStates(True)

        runType = RUN_TYPES[runTypeArg.pop()]
        if runType[TYPE] == 'all':
            params = runType[PARAMS]
            self.tables = TableMngr(NUM_ACTIONS, STATE_SIZE, runType[Q_TABLE], runType[RESULTS], params[0]) 
        
        elif runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            self.toProcessState = runType[PARAMS].isStateProcessed
            self.terminalStates = self.TerminalStates(self.toProcessState)
            self.tables = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[T_TABLE])     
         
        elif runType[TYPE] == 'hallucination':
            params = runType[PARAMS]
            self.tables = TableMngr(NUM_ACTIONS, STATE_SIZE, runType[Q_TABLE], runType[RESULTS], params[0], runType[T_TABLE], True)   
        elif runType[TYPE] == 'test':
            params = runType[PARAMS]
            self.tables = TestTableMngr(NUM_ACTIONS, runType[Q_TABLE], runType[RESULTS])
        elif runType[TYPE] == 'play':
            self.toProcessState = True
            self.tables = UserPlay()

        # states and action:
        self.current_action = None


        self.current_state = np.zeros((STATE_GRID_SIZE , STATE_GRID_SIZE), dtype=np.int32, order='C')
        self.previous_state = np.zeros((STATE_GRID_SIZE , STATE_GRID_SIZE), dtype=np.int32, order='C')
           
        self.selfLocCoord = []
        
    def step(self, obs):
        super(Agent, self).step(obs)
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
        return sc2Action
        r = float(obs.reward) + self.prevReward
        self.Learn(r)
        self.current_action = self.tables.choose_action(self.current_processed_state)
        time.sleep(STEP_DURATION)

        if self.current_action > ACTION_DO_NOTHING and self.selfLocCoord != None:   
            if self.IlligalMove():
                self.prevReward = REWARD_ILLIGAL_MOVE
            else:
                self.prevReward = 0

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

        self.current_state = np.zeros((STATE_GRID_SIZE , STATE_GRID_SIZE), dtype=np.int32, order='C')
        self.previous_state = np.zeros((STATE_GRID_SIZE , STATE_GRID_SIZE), dtype=np.int32, order='C')

        if self.toProcessState:
            self.current_processed_state = np.zeros(PROCESSED_STATE_SIZE, dtype=np.int32, order='C')
            self.previous_processed_state = np.zeros(PROCESSED_STATE_SIZE, dtype=np.int32, order='C')
        else:
            self.current_processed_state = self.current_state
            self.previous_processed_state = self.previous_state

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
            reward = 0
            s_ = self.terminalStates["tie"]

        if self.current_action is not None:
            self.tables.learn(self.current_processed_state.copy(), self.current_action, float(reward), s_, True)


        print("accumulated reward =", self.sumReward)
        self.tables.end_run(self.sumReward)

    def Learn(self, r):
        if self.current_action is not None:
            self.tables.learn(self.previous_processed_state.copy(), self.current_action, float(r), self.current_processed_state.copy(), r != 0)

        self.previous_state[:] = self.current_state[:]
        self.previous_processed_state[:] = self.current_processed_state[:]

    def CreateState(self, obs):
        self.current_state = np.zeros((STATE_GRID_SIZE , STATE_GRID_SIZE), dtype=np.int32, order='C')
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        
        selfLocX = []
        selfLocY = []
        for y in range(STATE_GRID_SIZE):
            for x in range(STATE_GRID_SIZE):
                self.current_state[y][x] = screenMap[y][x]
                if screenMap[y][x] == SC2_Params.PLAYER_SELF:
                    selfLocX.append(x)
                    selfLocY.append(y)
        
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

        if self.toProcessState:
            self.ProcessState()
        else:
            self.current_processed_state = self.current_state
    
    def ProcessState(self):
        self.current_processed_state = np.zeros(PROCESSED_STATE_SIZE, dtype=np.int32, order='C')

        selfMidLocX = self.selfLocCoord[SC2_Params.X_IDX]
        selfMidLocY = self.selfLocCoord[SC2_Params.Y_IDX]
        yScaled = int((selfMidLocY / STATE_GRID_SIZE) * PROCESSED_STATE_GRID_SIZE)
        xScaled = int((selfMidLocX / STATE_GRID_SIZE) * PROCESSED_STATE_GRID_SIZE)

        self.current_processed_state[SELF_MAT_START + xScaled + yScaled * PROCESSED_STATE_GRID_SIZE] = 1
        beacon_y, beacon_x = (self.current_state == SC2_Params.PLAYER_NEUTRAL).nonzero()
        if len(beacon_y) > 0:
            xBeaconMid = int(sum(beacon_x) / len(beacon_x))
            yBeaconMid = int(sum(beacon_y) / len(beacon_y))
            yScaled = int((yBeaconMid / STATE_GRID_SIZE) * PROCESSED_STATE_GRID_SIZE)
            xScaled = int((xBeaconMid / STATE_GRID_SIZE) * PROCESSED_STATE_GRID_SIZE)
            self.current_processed_state[BEACON_MAT_START + xScaled + yScaled * PROCESSED_STATE_GRID_SIZE] = 1


    def TerminalStates(self, toProcessState):
        tStates = {}
        if not toProcessState:
            state = np.zeros((STATE_GRID_SIZE , STATE_GRID_SIZE), dtype=np.int32, order='C')
            state[0,0] = -1
            tStates["win"] = state.copy()
            state[0,0] = -2
            tStates["tie"] = state.copy()
            state[0,0] = -3
            tStates["loss"] = state.copy()
        else:
            state = np.zeros(PROCESSED_STATE_SIZE, dtype=np.int32, order='C')
            state[0] = -1
            tStates["win"] = state.copy()
            state[0] = -2
            tStates["tie"] = state.copy()
            state[0] = -3
            tStates["loss"] = state.copy()

        return tStates
    def IlligalMove(self):
        change = GOTO_CHANGE[self.current_action]
        for i in range(2):
            if change[i] > 0:
                if self.selfLocCoord[i] == SCREEN_MAX[i]:
                    return True
            elif change[i] < 0:
                if self.selfLocCoord[i] == SCREEN_MIN[i]:
                    return True
        
        return False
        

        

