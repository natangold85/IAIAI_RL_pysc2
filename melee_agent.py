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

from utils import TableMngr
from utils import QTableParamsWOChangeInExploration
from utils import QTableParamsWithChangeInExploration

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp

WITH_EXPLORATION_CHANGE = True
SMART_EXPLORATION = 'smartExploration'
NAIVE_EXPLORATION = 'naiveExploration'
Q_TABLE_SMART_EXPLORATION = "melee_attack_qtable_smartExploration"
T_TABLE_SMART_EXPLORATION = "melee_attack_ttable_smartExploration"
R_TABLE_SMART_EXPLORATION = "melee_attack_rtable_smartExploration"
RESULT_SMART_EXPLORATION = "melee_attack_result_smartExploration"

Q_TABLE_NAIVE_EXPLORATION = "melee_attack_qtable_naiveExploration"
T_TABLE_NAIVE_EXPLORATION = "melee_attack_ttable_naiveExploration"
R_TABLE_NAIVE_EXPLORATION = "melee_attack_rtable_naiveExploration"
RESULT_NAIVE_EXPLORATION = "melee_attack_result_naiveExploration"


NUM_ENEMIES = 2

ID_ACTION_DO_NOTHING = 0
ID_ACTION_MOVE_NORTH = 1
ID_ACTION_MOVE_SOUTH = 2
ID_ACTION_MOVE_EAST = 3
ID_ACTION_MOVE_WEST = 4

NUM_MOVE_ACTIONS = 5

NUM_ACTIONS = NUM_MOVE_ACTIONS + NUM_ENEMIES

TO_MOVE = 4
MOVES_LUT = {}
MOVES_LUT[ID_ACTION_MOVE_NORTH] = [-TO_MOVE, 0]
MOVES_LUT[ID_ACTION_MOVE_SOUTH] = [TO_MOVE, 0]
MOVES_LUT[ID_ACTION_MOVE_WEST] = [0, -TO_MOVE]
MOVES_LUT[ID_ACTION_MOVE_EAST] = [0, +TO_MOVE]
# state details
STATE_NON_VALID_NUM = -1
STATE_GRID_SIZE = 10

STATE_SELF_LOCATION = 0
STATE_LAST_ATTACK_ACTION = 1
STATE_ENEMIES_START_IDX_LOCATION = 2

STATE_SIZE = STATE_ENEMIES_START_IDX_LOCATION + NUM_ENEMIES

ENEMY_ACTION_2_STATE = {}
for i in range (0, NUM_ENEMIES):
    ENEMY_ACTION_2_STATE[NUM_MOVE_ACTIONS + i] = STATE_ENEMIES_START_IDX_LOCATION + i

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

DEAD_UNIT = [-1,-1]
DEAD_UNIT_IDX = STATE_GRID_SIZE * STATE_GRID_SIZE

STEP_DURATION = 0

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

        # tables:
        if SMART_EXPLORATION in sys.argv:
            qTableParams = QTableParamsWithChangeInExploration()
            self.tables = TableMngr(NUM_ACTIONS, Q_TABLE_SMART_EXPLORATION, qTableParams, T_TABLE_SMART_EXPLORATION, RESULT_SMART_EXPLORATION)           
        elif NAIVE_EXPLORATION in sys.argv:
            qTableParams = QTableParamsWOChangeInExploration()
            self.tables = TableMngr(NUM_ACTIONS, Q_TABLE_NAIVE_EXPLORATION, qTableParams, T_TABLE_NAIVE_EXPLORATION, RESULT_NAIVE_EXPLORATION)
        else:
            print("Error: Enter typeof exploration!!")
            exit(1)            
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
        self.current_action = self.tables.choose_action(str(self.current_scaled_state))
        
        time.sleep(STEP_DURATION)
        sc2Action = DO_NOTHING_SC2_ACTION

        if IsAttackAction(self.current_action):
            coord2Attack = self.current_state[ENEMY_ACTION_2_STATE[self.current_action]]
            self.lastAttackAction = self.current_action
            if coord2Attack != DEAD_UNIT and SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(coord2Attack)])
        elif IsMoveAction(self.current_action):            
            goTo = []
            goTo[:] = self.current_state[STATE_SELF_LOCATION]

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
        
        self.lastAttackAction = ID_ACTION_DO_NOTHING

        self.errorOccur = False

    def LastStep(self, obs):
        if obs.reward > 0:
            reward = 1
        elif obs.reward < 0:
            reward = -1
        else:
            reward = -1

        self.tables.end_run(str(self.previous_scaled_state), self.current_action, reward)

    def Learn(self):
        if self.current_action is not None:
            self.tables.learn(str(self.previous_scaled_state), self.current_action, 0, str(self.current_scaled_state))

        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def CreateState(self, obs):
        if not self.m_initState:
            self.InitState(obs)
            self.m_initState = True
        else:
            self.current_state[STATE_SELF_LOCATION] = self.GetSelfLocation(obs)
            self.SetStateEnemyLocation(obs)
            self.current_state[STATE_LAST_ATTACK_ACTION] = self.lastAttackAction

        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[STATE_LAST_ATTACK_ACTION] = self.current_state[STATE_LAST_ATTACK_ACTION]
        self.current_scaled_state[STATE_SELF_LOCATION] = Coord2ScaledIdx(self.current_state[STATE_SELF_LOCATION])
        for i in range(STATE_ENEMIES_START_IDX_LOCATION, STATE_ENEMIES_START_IDX_LOCATION + NUM_ENEMIES):
            self.current_scaled_state[i] = Coord2ScaledIdx(self.current_state[i])


        
    def InitState(self, obs):
        self.current_state[STATE_SELF_LOCATION] = self.GetSelfLocation(obs)
        self.current_state[STATE_LAST_ATTACK_ACTION] = ID_ACTION_DO_NOTHING
        
        midPnt = self.GetEnemyLocation(obs)
        if len(midPnt) != NUM_ENEMIES:
            print("\n\n\n\nError in retirieved num enemies. stopping current trial")
            self.errorOccur = True
            for i in range(len(midPnt), NUM_ENEMIES):
                midPnt.append(DEAD_UNIT)

        midPnt.sort()
        for i in  range (0, NUM_ENEMIES):
            self.current_state[STATE_ENEMIES_START_IDX_LOCATION + i] = midPnt[i]
        
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
        unitSize1Axis1Directiom = int((TerranUnit.MARINE_SCREEN_SIZE_ONE_AXIS - 1) / 2)

        for idx in range(0, len(enemy_y)):
            if enemy_y[idx] - unitSize1Axis1Directiom >= 0 and enemy_y[idx] + unitSize1Axis1Directiom < SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX] and enemy_x[idx] - unitSize1Axis1Directiom >= 0 and enemy_x[idx] + unitSize1Axis1Directiom < SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]:
                
                if densityMat[enemy_y[idx]][enemy_x[idx]] > 1:
                    continue

                found = True
                for y in range(enemy_y[idx] - unitSize1Axis1Directiom, enemy_y[idx] + unitSize1Axis1Directiom + 1):
                    for x in range(enemy_x[idx] - unitSize1Axis1Directiom, enemy_x[idx] + unitSize1Axis1Directiom + 1):
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
        stateEnemiesOffset = STATE_ENEMIES_START_IDX_LOCATION

        if len(enemyPoints) == 0:
            self.AllEnemiesDead()
            return

        # add known dead enemies
        for e in self.m_deadEnemies:
            enemyPoints.append(DEAD_UNIT)

        newDead = len(enemyPoints) < NUM_ENEMIES
        # add new deads
        if newDead:
            if IsAttackAction(self.current_action):
                enemyPoints.append(DEAD_UNIT)
                self.m_deadEnemies.append(ENEMY_ACTION_2_STATE[self.current_action])
                self.current_state[ENEMY_ACTION_2_STATE[self.current_action]] = DEAD_UNIT

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
        for i in  range (STATE_ENEMIES_START_IDX_LOCATION, STATE_ENEMIES_START_IDX_LOCATION + NUM_ENEMIES):
            self.m_deadEnemies.append(i)
            self.current_state[i] = DEAD_UNIT

