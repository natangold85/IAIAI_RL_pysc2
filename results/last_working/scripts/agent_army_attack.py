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

#decision makers
from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints

TIME_LINE_BUCKETING = 25
NON_VALID_LOCATION_QVAL = -2

def NumActions(gridSize):
    return gridSize * gridSize + 1
def NumStateLocationsVal(gridSize):
    return 2 * gridSize * gridSize
def NumStateVal(gridSize):
    return 2 * gridSize * gridSize + 1

def NNFunc_2Layers(x, numActions, scope):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        output = tf.contrib.layers.fully_connected(fc2, numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
    return output

    # Define the neural network
def build_dqn_0init(x, numActions, scope):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512, activation_fn = tf.nn.softplus, weights_initializer=tf.zeros_initializer())
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid, weights_initializer=tf.zeros_initializer()) * 2 - 1
    return output

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
QTABLE_GS5 = 'qGS5'
DQN_GS5_ARMY = 'dqnGS5'
DQN_GS5_EMBEDDING_LOCATIONS = 'dqnGS5_Embedding' 
DQN_GS5_2LAYERS = 'dqnGS5_2Layers' 

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE_GS5, DQN_GS5_ARMY, DQN_GS5_EMBEDDING_LOCATIONS, DQN_GS5_2LAYERS])

def BaseTestSet(gridSize):    
    testSet = []

    s1 = np.zeros(2 * gridSize * gridSize + 1, dtype = int)
    
    s1[5] = 5
    s1[gridSize * gridSize + 8] = 1
    s1[gridSize * gridSize + 9] = 1
    s1[gridSize * gridSize + 13] = 1
    s1[2 * gridSize * gridSize] = gridSize * gridSize
    actions_s1 = [8, 9, 13, 14, gridSize * gridSize]

    testSet.append([s1, actions_s1])

    s2 = np.zeros(2 * gridSize * gridSize + 1, dtype = int)

    s2[5] = 2
    s2[gridSize * gridSize + 8] = 2
    s2[gridSize * gridSize + 14] = 3
    s2[2 * gridSize * gridSize] = gridSize * gridSize
    actions_s2 = [8, 14, gridSize * gridSize]

    testSet.append([s2, actions_s2])

    s3 = np.zeros(2 * gridSize * gridSize + 1, dtype = int)
    s3[11] = 5
    s3[gridSize * gridSize + 12] = 2
    s3[gridSize * gridSize + 23] = 2
    actions_s3 = [12, 13, 23, gridSize * gridSize]

    testSet.append([s3, actions_s3])

    return testSet

# table type
TYPE = "type"
NN = "nn"
Q_TABLE = "q"
T_TABLE = "t"
HISTORY = "hist"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'
GRIDSIZE_key = 'gridsize'
BUILDING_VALUES_key = 'buildingValues'

# table names
ARMY_ATTACK_RUN_TYPES = {}


ARMY_ATTACK_RUN_TYPES[QTABLE_GS5] = {}
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][TYPE] = "QReplay"
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][GRIDSIZE_key] = 5
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][PARAMS] = QTableParamsExplorationDecay(NumStateVal(5), NumActions(5))
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][NN] = ""
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][Q_TABLE] = "armyBattle_qGS5_qtable"
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][HISTORY] = "armyBattle_qGS5_replayHistory"
ARMY_ATTACK_RUN_TYPES[QTABLE_GS5][RESULTS] = "armyBattle_qGS5_result"

ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY] = {}
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][TYPE] = "NN"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][GRIDSIZE_key] = 5
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5))
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][NN] = "armyBattle_dqnGS5_DQN"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][Q_TABLE] = ""
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][HISTORY] = "armyBattle_dqnGS5_replayHistory"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_ARMY][RESULTS] = "armyBattle_dqnGS5_result"

ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS] = {}
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][TYPE] = "NN"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][GRIDSIZE_key] = 5
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(NumStateVal(5), NumStateLocationsVal(5), NumActions(5))
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][NN] = "armyBattle_dqnGS5_Embedding_DQN"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][Q_TABLE] = ""
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][HISTORY] = "armyBattle_dqnGS5_Embedding_replayHistory"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][RESULTS] = "armyBattle_dqnGS5_Embedding_result"

ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS] = {}
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][TYPE] = "NN"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][GRIDSIZE_key] = 5
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][PARAMS] = DQN_PARAMS(NumStateVal(5), NumActions(5), nn_Func = NNFunc_2Layers)
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][NN] = "armyBattle_dqnGS5_2Layers_DQN"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][Q_TABLE] = ""
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][HISTORY] = "armyBattle_dqnGS5_2Layers_replayHistory"
ARMY_ATTACK_RUN_TYPES[DQN_GS5_2LAYERS][RESULTS] = "armyBattle_dqnGS5_2Layers_result"


ARMY_ATTACK_RUN_TYPES[USER_PLAY] = {}
ARMY_ATTACK_RUN_TYPES[USER_PLAY][GRIDSIZE_key] = 5
ARMY_ATTACK_RUN_TYPES[USER_PLAY][TYPE] = "play"
    
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



class ArmyAttack(base_agent.BaseAgent):
    def __init__(self, mainAgent = True, runArg = '', tfSession = None):        
        super(ArmyAttack, self).__init__()

        self.mainAgent = mainAgent

        if runArg == '':
            runTypeArg = ALL_TYPES.intersection(sys.argv)
            if len(runTypeArg) != 1:
                print("\n\nplay type not entered correctly\n\n")
                exit() 

            runArg = runTypeArg.pop()

        runType = ARMY_ATTACK_RUN_TYPES[runArg]

        # state and actions:

        self.current_action = None
        self.gridSize = runType[GRIDSIZE_key]
        self.numActions = self.gridSize * self.gridSize + 1

        self.state_startSelfMat = 0
        self.state_startEnemyMat = self.gridSize * self.gridSize
        self.state_timeLineIdx = 2 * self.gridSize * self.gridSize

        self.state_size = 2 * self.gridSize * self.gridSize + 1
        
        self.terminalState = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.doNtohingStop = True
        self.illigalmoveSolveInModel = True
        self.illigalMoveReward = 0

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
            runType[PARAMS].tfSession = tfSession
            self.decisionMaker = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            self.doNtohingAction = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])
            self.decisionMaker = UserPlay(False, self.numActions, self.gridSize * self.gridSize)
        else:
            print("\n\ninvalid run type\n\n")
            exit()

        self.lastValidAttackAction = None
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.enemyBuildingGridLoc2ScreenLoc = {}


    def step(self, obs):
        super(ArmyAttack, self).step(obs)
        
        if obs.first():
            self.FirstStep(obs)
        elif obs.last():
            self.LastStep(obs)
            return SC2_Actions.DO_NOTHING_SC2_ACTION
            

        self.CreateState(obs)
        if self.mainAgent:
            self.Learn()
            time.sleep(STEP_DURATION)
        
        self.current_action = self.ChooseAction()
        self.numStep += 1
        
        # print("\n")
        # self.PrintState()
        # print(self.enemyArmyGridLoc2ScreenLoc)

        if self.mainAgent:
            return self.Action2SC2Action(obs, self.current_action)
        else:
            return self.current_action

    def FirstStep(self, obs):
        self.numStep = 0

        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.current_action = None
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.selfLocCoord = None      
        self.errorOccur = False

        self.prevReward = 0

        return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])

    def LastStep(self, obs, reward = 0):

        if self.mainAgent:
            if obs.reward > 0:
                reward = 1.0
            elif obs.reward < 0:
                reward = -1.0
            else:
                reward = -1.0

            if self.current_action is not None:
                self.decisionMaker.learn(self.current_state.copy(), self.current_action, float(reward), self.terminalState.copy(), True)

            score = obs.observation["score_cumulative"][0]
            self.decisionMaker.end_run(reward, score, self.numStep)
    
    def Learn(self):
        if self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, float(self.prevReward), self.current_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.prevReward = 0.0

    def Action2SC2Action(self, obs, a):
        if SC2_Actions.STOP in obs.observation['available_actions']:
            sc2Action = SC2_Actions.STOP_SC2_ACTION
        else:
            sc2Action = SC2_Actions.DO_NOTHING_SC2_ACTION

        if self.current_action < self.gridSize * self.gridSize:     
            goTo = self.enemyArmyGridLoc2ScreenLoc[self.current_action].copy()
            if SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
       
        return sc2Action

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