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

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_tables import TableMngr
from utils_tables import TestTableMngr
from utils_tables import LearnWithReplayMngr
from utils_tables import UserPlay

from utils_tables import QTableParamsWOChangeInExploration
from utils_tables import QTableParamsWithChangeInExploration
from utils_tables import QTablePropogation

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp

# possible types of play
SMART_EXPLORATION_GRID_SIZE_2 = 'smartExploration'
NAIVE_EXPLORATION_GRID_SIZE_2 = 'naiveExploration'
DQN_CMP_QTABLE = "dqnCmpQ"
DQN_CMP_QTABLE_FULL = "dqnCmpQFull"
DQN_CMP_NEURAL_NETWORK = "dqnCmpNN"
DQN_CMP_NEURAL_NETWORK_FULL = "dqnCmpNNFull"
ONLINE_HALLUCINATION = 'onlineHallucination'
REWARD_PROPAGATION = 'rewardPropogation'

USER_PLAY = 'play'

ALL_TYPES = set([SMART_EXPLORATION_GRID_SIZE_2, NAIVE_EXPLORATION_GRID_SIZE_2, 
            ONLINE_HALLUCINATION, REWARD_PROPAGATION, USER_PLAY,
            DQN_CMP_QTABLE, DQN_CMP_NEURAL_NETWORK, DQN_CMP_QTABLE_FULL, DQN_CMP_NEURAL_NETWORK_FULL])

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

RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][PARAMS] = 1
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][NN] = "./melee_attack_DQN_Full"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][Q_TABLE] = "melee_attack_qtable_dqnCmpNN_Full"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][T_TABLE] = "melee_attack_history_dqnCmpNN_Full"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][R_TABLE] = ""
RUN_TYPES[DQN_CMP_NEURAL_NETWORK_FULL][RESULTS] = "melee_attack_result_dqnCmpNN_Full"

RUN_TYPES[DQN_CMP_QTABLE] = {}
RUN_TYPES[DQN_CMP_QTABLE][TYPE] = "NN"
RUN_TYPES[DQN_CMP_QTABLE][PARAMS] = 0.25
RUN_TYPES[DQN_CMP_QTABLE][NN] = ""
RUN_TYPES[DQN_CMP_QTABLE][Q_TABLE] = "melee_attack_qtable_dqnCmpQ"
RUN_TYPES[DQN_CMP_QTABLE][T_TABLE] = "melee_attack_history_dqnCmpQ"
RUN_TYPES[DQN_CMP_QTABLE][RESULTS] = "melee_attack_result_dqnCmpQ"

RUN_TYPES[DQN_CMP_QTABLE_FULL] = {}
RUN_TYPES[DQN_CMP_QTABLE_FULL][TYPE] = "NN"
RUN_TYPES[DQN_CMP_QTABLE_FULL][PARAMS] = 1
RUN_TYPES[DQN_CMP_QTABLE_FULL][NN] = ""
RUN_TYPES[DQN_CMP_QTABLE_FULL][Q_TABLE] = "melee_attack_qtable_dqnCmpQ_Full"
RUN_TYPES[DQN_CMP_QTABLE_FULL][T_TABLE] = "melee_attack_history_dqnCmpQ_Full"
RUN_TYPES[DQN_CMP_QTABLE_FULL][RESULTS] = "melee_attack_result_dqnCmpQ_Full"


RUN_TYPES[DQN_CMP_NEURAL_NETWORK] = {}
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][TYPE] = "NN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][PARAMS] = 0.25
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][NN] = "./melee_attack_DQN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][Q_TABLE] = "melee_attack_qtable_dqnCmpNN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][T_TABLE] = "melee_attack_history_dqnCmpNN"
RUN_TYPES[DQN_CMP_NEURAL_NETWORK][RESULTS] = "melee_attack_result_dqnCmpNN"


RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2] = {}
RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2][TYPE] = "all"
RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2][PARAMS] = [QTableParamsWithChangeInExploration()]
RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2][Q_TABLE] = "melee_attack_qtable_smartExploration"
RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2][T_TABLE] = ""
RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2][R_TABLE] = ""
RUN_TYPES[SMART_EXPLORATION_GRID_SIZE_2][RESULTS] = "melee_attack_result_smartExploration"

RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2] = {}
RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2][TYPE] = "all"
RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2][PARAMS] = [QTableParamsWOChangeInExploration()]
RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2][Q_TABLE] = "melee_attack_qtable_naiveExploration"
RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2][T_TABLE] = ""
RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2][R_TABLE] = ""
RUN_TYPES[NAIVE_EXPLORATION_GRID_SIZE_2][RESULTS] = "melee_attack_result_naiveExploration"

RUN_TYPES[ONLINE_HALLUCINATION] = {}
RUN_TYPES[ONLINE_HALLUCINATION][TYPE] = "hallucination"
RUN_TYPES[ONLINE_HALLUCINATION][PARAMS] = [QTableParamsWOChangeInExploration()]
RUN_TYPES[ONLINE_HALLUCINATION][Q_TABLE] = "melee_attack_qtable_onlineHallucination"
RUN_TYPES[ONLINE_HALLUCINATION][T_TABLE] = "melee_attack_ttable_onlineHallucination"
RUN_TYPES[ONLINE_HALLUCINATION][R_TABLE] = ""
RUN_TYPES[ONLINE_HALLUCINATION][RESULTS] = "melee_attack_result_onlineHallucination"

RUN_TYPES[REWARD_PROPAGATION] = {}
RUN_TYPES[REWARD_PROPAGATION][TYPE] = "all"
RUN_TYPES[REWARD_PROPAGATION][PARAMS] = [QTablePropogation()]
RUN_TYPES[REWARD_PROPAGATION][Q_TABLE] = "melee_attack_qtable_rewardPropogation"
RUN_TYPES[REWARD_PROPAGATION][T_TABLE] = ""
RUN_TYPES[REWARD_PROPAGATION][R_TABLE] = ""
RUN_TYPES[REWARD_PROPAGATION][RESULTS] = "melee_attack_result_rewardPropogation"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"


NUM_ENEMIES = 2

ID_ACTION_DO_NOTHING = 0
ID_ACTION_MOVE_NORTH = 1
ID_ACTION_MOVE_SOUTH = 2
ID_ACTION_MOVE_EAST = 3
ID_ACTION_MOVE_WEST = 4

NUM_MOVE_ACTIONS = 5

NUM_ACTIONS = NUM_MOVE_ACTIONS + NUM_ENEMIES

ATTACK_ACTION_2_ENEMY_NUM = {}
idx = 1
ATTACK_ACTION_2_ENEMY_NUM[ID_ACTION_DO_NOTHING] = 0
for i in range(NUM_MOVE_ACTIONS, NUM_ACTIONS):
    ATTACK_ACTION_2_ENEMY_NUM[i] = idx
    idx += 1

TO_MOVE = 4
MOVES_LUT = {}
MOVES_LUT[ID_ACTION_MOVE_NORTH] = [-TO_MOVE, 0]
MOVES_LUT[ID_ACTION_MOVE_SOUTH] = [TO_MOVE, 0]
MOVES_LUT[ID_ACTION_MOVE_WEST] = [0, -TO_MOVE]
MOVES_LUT[ID_ACTION_MOVE_EAST] = [0, +TO_MOVE]

# state details
STATE_NON_VALID_NUM = -1
STATE_GRID_SIZE = 10

RAW_STATE_SELF_LOC = 0
RAW_STATE_ENEMY1_LOC = 1
RAW_STATE_ENEMY2_LOC = 2
RAW_STATE_LAST_ATTACK = 3
RAW_STATE_SIZE = 4

STATE_SELF_LOCATION_MAT_START = 0
STATE_ENEMY_LOCATION_MAT_START = STATE_GRID_SIZE * STATE_GRID_SIZE

STATE_LAST_ATTACK_ENEMY = 2 * STATE_GRID_SIZE * STATE_GRID_SIZE
STATE_TIME_LINE = STATE_LAST_ATTACK_ENEMY + 1
STATE_SIZE = STATE_TIME_LINE + 1

TIME_LINE_BUCKETING = 25

ENEMY_ACTION_2_RAW_STATE = {}
for i in range (0, NUM_ENEMIES):
    ENEMY_ACTION_2_RAW_STATE[NUM_MOVE_ACTIONS + i] = RAW_STATE_ENEMY1_LOC + i

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

DEAD_UNIT = [-1,-1]
DEAD_UNIT_IDX = STATE_GRID_SIZE * STATE_GRID_SIZE

STEP_DURATION = 0
MARINE_SCREEN_SIZE_ONE_AXIS = 3

def IsAttackAction(action):
    return action >= NUM_MOVE_ACTIONS and action < NUM_MOVE_ACTIONS + NUM_ENEMIES

def IsMoveAction(action):
    return action > ID_ACTION_DO_NOTHING and action < NUM_MOVE_ACTIONS

def maxDiff(p1, p2):
    if p1 == DEAD_UNIT or p2 == DEAD_UNIT:
        return 0
    diffX = p1[SC2_Params.X_IDX] - p2[SC2_Params.X_IDX]
    diffY = p1[SC2_Params.Y_IDX] - p2[SC2_Params.Y_IDX]

    return max(abs(diffX), abs(diffY))

def FindSingleUnit(unitMat, point, unit_size):
    # first find left upper edge
    y = point[SC2_Params.Y_IDX]
    x = point[SC2_Params.X_IDX]

    for i in range(1, unit_size):
        if y - 1 >= 0:
            foundEdge = False
            if x - 1 >= 0:
                if unitMat[y - 1][x - 1]:
                    foundEdge = True
                    y -= 1
                    x -= 1
            if not foundEdge:
                if unitMat[y - 1][x]:
                    y -= 1

    # insert all points to vector
    pnts_y = []
    pnts_x = []
    for x_loc in range(0, unit_size):
        for y_loc in range(0, unit_size):
            pnts_y.append(y + y_loc)
            pnts_x.append(x + x_loc)

    return pnts_y, pnts_x

def Coord2ScaledIdx(point):
    if point == DEAD_UNIT:
        return DEAD_UNIT_IDX

    x = int(point[SC2_Params.X_IDX] * STATE_GRID_SIZE / SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX])
    y = int(point[SC2_Params.Y_IDX] * STATE_GRID_SIZE / SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX])

    return x + y * STATE_GRID_SIZE

def PrintScaledState(state):

    for y in range(0, STATE_GRID_SIZE):
        for x in range(0, STATE_GRID_SIZE):
            idx = 0
            objLoc = False
            for loc in state[:]:
                if x + y * STATE_GRID_SIZE == loc:
                    objLoc = True
                    break
                idx += 1

            if objLoc:
                print(idx, end = '')
            else:
                print('_', end = '')
        print('|')

class Attack(base_agent.BaseAgent):
    def __init__(self):        
        super(Attack, self).__init__()

        runTypeArg = ALL_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("play type not entered correctly")
            exit(1) 

        runType = RUN_TYPES[runTypeArg.pop()]
        if runType[TYPE] == 'all':
            params = runType[PARAMS]
            self.tables = TableMngr(NUM_ACTIONS, STATE_SIZE, runType[Q_TABLE], runType[RESULTS], params[0]) 
        
        if runType[TYPE] == 'NN':
            self.terminalStates = {}
            state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
            state[0] = -1
            self.terminalStates["win"] = state.copy()
            state[0] = -2
            self.terminalStates["tie"] = state.copy()
            state[0] = -3
            self.terminalStates["loss"] = state.copy()
            self.tables = LearnWithReplayMngr(NUM_ACTIONS, STATE_SIZE, runType[NN] != '', self.terminalStates, runType[NN], runType[Q_TABLE], runType[RESULTS], runType[T_TABLE], runType[PARAMS])     
        
        elif runType[TYPE] == 'hallucination':
            params = runType[PARAMS]
            self.tables = TableMngr(NUM_ACTIONS, STATE_SIZE, runType[Q_TABLE], runType[RESULTS], params[0], runType[T_TABLE], True)   
        elif runType[TYPE] == 'test':
            params = runType[PARAMS]
            self.tables = TestTableMngr(NUM_ACTIONS, runType[Q_TABLE], runType[RESULTS])
        elif runType[TYPE] == 'play':
            self.tables = UserPlay()

        # states and action:
        self.current_action = None

        self.current_state = []
        self.previous_state = []


        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        # model params
        self.m_initState = False
        self.m_deadEnemies = []
        self.lastAttackAction = ID_ACTION_DO_NOTHING

    def step(self, obs):
        super(Attack, self).step(obs)
        if obs.first():
            self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
        
        self.CreateState(obs)
        if self.errorOccur:
            return DO_NOTHING_SC2_ACTION

        self.numStep += 1
        self.Learn()
        self.current_action = self.tables.choose_action(self.current_scaled_state)
        
        time.sleep(STEP_DURATION)
        sc2Action = DO_NOTHING_SC2_ACTION

        if IsAttackAction(self.current_action):
            coord2Attack = self.current_state[ENEMY_ACTION_2_RAW_STATE[self.current_action]]
            self.lastAttackAction = self.current_action
            if coord2Attack != DEAD_UNIT and SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(coord2Attack)])
        elif IsMoveAction(self.current_action):            
            goTo = []
            goTo[:] = self.current_state[RAW_STATE_SELF_LOC]

            goTo[SC2_Params.X_IDX] += MOVES_LUT[self.current_action][SC2_Params.X_IDX]
            goTo[SC2_Params.Y_IDX] += MOVES_LUT[self.current_action][SC2_Params.Y_IDX]
                   
            if goTo[0] >= 0 and goTo[1] >= 0 and goTo[0] < SC2_Params.SCREEN_SIZE[0] and goTo[1] < SC2_Params.SCREEN_SIZE[1]:
                if SC2_Actions.MOVE_IN_SCREEN in obs.observation['available_actions']:
                    sc2Action = actions.FunctionCall(SC2_Actions.MOVE_IN_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])


        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0

        self.m_initState = False
        self.m_deadEnemies = []
        
        self.current_state = []
        for i in range (0, STATE_SIZE):
            self.current_state.append(0)

        self.previous_state = []
        for i in range (0, STATE_SIZE):
            self.previous_state.append(0)

        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
                
        self.lastAttackAction = ID_ACTION_DO_NOTHING

        self.errorOccur = False

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
            self.tables.learn(self.current_scaled_state.copy(), self.current_action, reward, s_)

        self.tables.end_run(reward)

    def Learn(self):
        if self.current_action is not None:
            self.tables.learn(self.previous_scaled_state.copy(), self.current_action, 0, self.current_scaled_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def CreateState(self, obs):
        if not self.m_initState:
            self.InitState(obs)
            self.m_initState = True
        else:
            self.current_state[RAW_STATE_SELF_LOC] = self.GetSelfLocation(obs)
            self.SetStateEnemyLocation(obs)
            self.current_state[RAW_STATE_LAST_ATTACK] = self.lastAttackAction

        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[STATE_LAST_ATTACK_ENEMY] = ATTACK_ACTION_2_ENEMY_NUM[self.current_state[RAW_STATE_LAST_ATTACK]]
        for i in range(STATE_GRID_SIZE * STATE_GRID_SIZE):
            if Coord2ScaledIdx(self.current_state[RAW_STATE_SELF_LOC]) == i:
                self.current_scaled_state[STATE_SELF_LOCATION_MAT_START + i] = 1
            else:
                self.current_scaled_state[STATE_SELF_LOCATION_MAT_START + i] = 0

            toInsertToRedMat = 0
            if Coord2ScaledIdx(self.current_state[RAW_STATE_ENEMY1_LOC]) == i:
                toInsertToRedMat += 1
            if Coord2ScaledIdx(self.current_state[RAW_STATE_ENEMY2_LOC]) == i:
                toInsertToRedMat += 2

            self.current_scaled_state[STATE_ENEMY_LOCATION_MAT_START + i] = toInsertToRedMat

        self.current_scaled_state[STATE_TIME_LINE] = int(self.numStep / TIME_LINE_BUCKETING)


        
    def InitState(self, obs):
        self.current_state[RAW_STATE_SELF_LOC] = self.GetSelfLocation(obs)
        self.current_state[RAW_STATE_LAST_ATTACK] = ID_ACTION_DO_NOTHING
        
        midPnt = self.GetEnemyLocation(obs)
        if len(midPnt) != NUM_ENEMIES:
            print("\n\n\n\nError in retirieved num enemies. stopping current trial")
            self.errorOccur = True
            for i in range(len(midPnt), NUM_ENEMIES):
                midPnt.append(DEAD_UNIT)

        midPnt.sort()
        for i in  range (0, NUM_ENEMIES):
            self.current_state[RAW_STATE_ENEMY1_LOC + i] = midPnt[i]
        
    def GetSelfLocation(self, obs):
        self_y, self_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
        if len(self_y) > 0:
            return FindMiddle(self_y, self_x)
        else:
            return DEAD_UNIT

    def GetEnemyLocation(self, obs):
        enemyMat = obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE
        enemy_y, enemy_x = enemyMat.nonzero()
        
        densityMat = obs.observation['screen'][SC2_Params.UNIT_DENSITY]
        enemiesMidPoint = []
        unitSize1Axis1Direction = int((MARINE_SCREEN_SIZE_ONE_AXIS - 1) / 2)

        for idx in range(0, len(enemy_y)):
            if enemy_y[idx] - unitSize1Axis1Direction >= 0 and enemy_y[idx] + unitSize1Axis1Direction < SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] and enemy_x[idx] - unitSize1Axis1Direction >= 0 and enemy_x[idx] + unitSize1Axis1Direction < SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]:
                
                if densityMat[enemy_y[idx]][enemy_x[idx]] > 1:
                    continue

                found = True
                for y in range(enemy_y[idx] - unitSize1Axis1Direction, enemy_y[idx] + unitSize1Axis1Direction + 1):
                    for x in range(enemy_x[idx] - unitSize1Axis1Direction, enemy_x[idx] + unitSize1Axis1Direction + 1):
                        if not enemyMat[y][x]:
                            found = False

                if found:
                    enemiesMidPoint.append([enemy_y[idx], enemy_x[idx]])

        enemies = []
        enemies[:] = enemiesMidPoint[:]
        for e1 in enemiesMidPoint[:]:
            countClose0Axis = 0
            countClose1Axis = 0
            for e2 in enemiesMidPoint[:]:
                if e1[1] == e2[1]:
                    if e1[0] + 1 == e2[0] or e1[0] - 1 == e2[0]:
                        countClose0Axis += 1
                if e1[0] == e2[0]:
                    if e1[1] + 1 == e2[1] or e1[1] - 1 == e2[1]:
                        countClose1Axis += 1

            if countClose0Axis > 1 or countClose1Axis > 1:
                enemies.remove(e1)
            
        if len(enemies) > NUM_ENEMIES :
            # something wasn't right return error
            return [-1,-1]

        return enemies


    def SetStateEnemyLocation(self, obs):
        enemyPoints = self.GetEnemyLocation(obs)
        stateEnemiesOffset = RAW_STATE_ENEMY1_LOC

        if len(enemyPoints) == 0:
            self.AllEnemiesDead()
            return

        # add known dead enemies
        for e in self.m_deadEnemies:
            enemyPoints.append(DEAD_UNIT)

        newDead = len(enemyPoints) < NUM_ENEMIES
        # add new deads
        if newDead:
            while len(enemyPoints) != NUM_ENEMIES:
                enemyPoints.append(DEAD_UNIT)

        distMat = []
        reviveEnemyValue = 10000
        for inMap in range(0, NUM_ENEMIES):
            distMat.append([])
            numStateLiving = 0
            for inState in range(stateEnemiesOffset, stateEnemiesOffset + NUM_ENEMIES):
                if enemyPoints[inMap] == DEAD_UNIT:
                    distMat[inMap].append(0)
                elif self.current_state[inState] == DEAD_UNIT:
                    distMat[inMap].append(reviveEnemyValue)
                else:
                    dist = DistForCmp(enemyPoints[inMap], self.current_state[inState])
                    distMat[inMap].append(dist)
                numStateLiving += 1

        idx2Include = []
        for i in range(0, NUM_ENEMIES):
            idx2Include.append(-1)
        
        minDist = 10000000
        idxMap2State, minDist = self.MatchLocation2State(distMat, idx2Include, minDist, 0, 0)
        for idx in range(0, NUM_ENEMIES):
            enemyStateIdx = idxMap2State[idx] + stateEnemiesOffset         
            self.current_state[enemyStateIdx] = enemyPoints[idx]


        self.m_deadEnemies = []
        for e in range(stateEnemiesOffset, stateEnemiesOffset + NUM_ENEMIES):
            if self.current_state[e] == DEAD_UNIT:
                self.m_deadEnemies.append(e)

        
    def MatchLocation2State(self, distMat, idxIncluded, minSumDist, currSumDist, currPoint):
        if currPoint == NUM_ENEMIES:
            if currSumDist < minSumDist:
                return idxIncluded[:], currSumDist
            else:
                return [], minSumDist
        else:
            minIdxVec = []
            for pnt in range(0, NUM_ENEMIES):
                if pnt not in idxIncluded:
                    idxIncluded[currPoint] = pnt
                    idxVec, minSumDist = self.MatchLocation2State(distMat, idxIncluded, minSumDist, currSumDist + distMat[currPoint][pnt], currPoint + 1)
                    idxIncluded[currPoint] = -1
                    if idxVec != []:
                        minIdxVec = idxVec

            return minIdxVec, minSumDist

    def AllEnemiesDead(self):
        self.m_deadEnemies = []
        for i in  range (RAW_STATE_ENEMY1_LOC, RAW_STATE_ENEMY1_LOC + NUM_ENEMIES):
            self.m_deadEnemies.append(i)
            self.current_state[i] = DEAD_UNIT

