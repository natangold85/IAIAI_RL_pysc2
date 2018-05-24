# build base sub agent
import sys
import random
import math
import time
import os.path
import datetime

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp

from utils_tables import UserPlay
from utils_tables import TableMngr
from utils_tables import QTableParamsWOChangeInExploration
from utils_tables import QTableParamsWithChangeInExploration

SMART_EXPLORATION_GRID_SIZE_2 = 'smartExploration2'
NAIVE_EXPLORATION_GRID_SIZE_2 = 'naiveExploration2'
NAIVE_EXPLORATION_GRID_SIZE_4 = 'naiveExploration4'
USER_PLAY = 'play'


Q_TABLE_SMART_EXPLORATION = "melee_attack_2_qtable_smartExploration"
T_TABLE_SMART_EXPLORATION = "melee_attack_2_ttable_smartExploration"
R_TABLE_SMART_EXPLORATION = "melee_attack_2_rtable_smartExploration"
RESULT_SMART_EXPLORATION = "melee_attack_2_result_smartExploration"

Q_TABLE_NAIVE_EXPLORATION = "melee_attack_2_qtable_naiveExploration"
T_TABLE_NAIVE_EXPLORATION = "melee_attack_2_ttable_naiveExploration"
R_TABLE_NAIVE_EXPLORATION = "melee_attack_2_rtable_naiveExploration"
RESULT_NAIVE_EXPLORATION = "melee_attack_2_result_naiveExploration"

Q_TABLE_NAIVE_EXPLORATION4 = "melee_attack_4_qtable_naiveExploration"
T_TABLE_NAIVE_EXPLORATION4 = "melee_attack_4_ttable_naiveExploration"
R_TABLE_NAIVE_EXPLORATION4 = "melee_attack_4_rtable_naiveExploration"
RESULT_NAIVE_EXPLORATION4 = "melee_attack_4_result_naiveExploration"



NON_VALID_NUM = -1

class STATE:
    GRID_SIZE = 2

    # non-scaled state details
    SIZE = 3 * GRID_SIZE * GRID_SIZE
    SCALED_SIZE = 2 * GRID_SIZE * GRID_SIZE + 2

    STEP_BUCKETING = 30 

    SELF_START_IDX = 0
    ENEMY_START_IDX = GRID_SIZE * GRID_SIZE
    SELECTED_START_IDX = 2 * GRID_SIZE * GRID_SIZE
    STEP_IDX = 2 * GRID_SIZE * GRID_SIZE + 1

    POWER_BUCKETING = 1

    SCALED_NON_SELECT = 0

class ACTIONS:
    DO_NOTHING = 1
    SELECT_ALL = DO_NOTHING + 1
    SELECT = STATE.GRID_SIZE * STATE.GRID_SIZE + SELECT_ALL
    MOVE = STATE.GRID_SIZE * STATE.GRID_SIZE + SELECT
    NUM_ACTIONS = MOVE


DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

REWARD_ILLIGAL_MOVE = -1

STEP_DURATION = 0

UNIT_IN_MAP = 0
NUM_UNIT_SCREEN_PIXELS = 0

def changeModelParams(newGridSize, powerBucketing):
    STATE.GRID_SIZE = newGridSize
    ACTIONS.SELECT = STATE.GRID_SIZE * STATE.GRID_SIZE + ACTIONS.SELECT_ALL
    ACTIONS.MOVE = STATE.GRID_SIZE * STATE.GRID_SIZE + ACTIONS.SELECT
    ACTIONS.NUM_ACTIONS = ACTIONS.MOVE

    STATE.SIZE = 3 * STATE.GRID_SIZE * STATE.GRID_SIZE
    STATE.SCALED_SIZE = 2 * STATE.GRID_SIZE * STATE.GRID_SIZE + 2

    STATE.SELF_START_IDX = 0
    STATE.ENEMY_START_IDX = STATE.GRID_SIZE * STATE.GRID_SIZE
    STATE.SELECTED_START_IDX = 2 * STATE.GRID_SIZE * STATE.GRID_SIZE
    STATE.STEP_IDX = 2 * STATE.GRID_SIZE * STATE.GRID_SIZE + 1

    STATE.POWER_BUCKETING = powerBucketing

for key,value in TerranUnit.UNIT_SPEC.items():
    if value.name == "marine":
        UNIT_IN_MAP = key
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels

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


def ScaleScreenPoint(y, x, screenStart, screenSize, newGridSize):
    y -= screenStart[SC2_Params.Y_IDX]
    x -= screenStart[SC2_Params.X_IDX]

    y /= screenSize[SC2_Params.Y_IDX]
    x /= screenSize[SC2_Params.X_IDX]

    y *= newGridSize
    x *= newGridSize    

    return int(y), int(x)

class Attack(base_agent.BaseAgent):
    def __init__(self):        
        super(Attack, self).__init__()

        # tables:
        if SMART_EXPLORATION_GRID_SIZE_2 in sys.argv:
            changeModelParams(2, 4)
            qTableParams = QTableParamsWithChangeInExploration()
            self.tables = TableMngr(ACTIONS.NUM_ACTIONS, Q_TABLE_SMART_EXPLORATION, qTableParams, T_TABLE_SMART_EXPLORATION, RESULT_SMART_EXPLORATION)           
        elif NAIVE_EXPLORATION_GRID_SIZE_2 in sys.argv:
            changeModelParams(2, 4)
            qTableParams = QTableParamsWOChangeInExploration()
            self.tables = TableMngr(ACTIONS.NUM_ACTIONS, Q_TABLE_NAIVE_EXPLORATION, qTableParams,T_TABLE_NAIVE_EXPLORATION, RESULT_NAIVE_EXPLORATION)
        elif NAIVE_EXPLORATION_GRID_SIZE_4 in sys.argv:
            changeModelParams(4, 4)
            qTableParams = QTableParamsWOChangeInExploration()
            self.tables = TableMngr(ACTIONS.NUM_ACTIONS, Q_TABLE_NAIVE_EXPLORATION4, qTableParams, T_TABLE_NAIVE_EXPLORATION4, RESULT_NAIVE_EXPLORATION4)
        elif USER_PLAY in sys.argv:
            self.tables = UserPlay()
        else:
            print("Error: Enter typeof exploration!!")
            exit(1) 
               
    
        # states and action:
        self.current_action = None

        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        
        self.current_scaled_state = np.zeros(STATE.SCALED_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SCALED_SIZE, dtype=np.int32, order='C')

        # model params
        self.screenStart = [0,0]
        self.screenSize = [0,0]

        self.regionSize = [0,0]
        self.coordStartLocation = []

        self.startTime = None

    def step(self, obs):
        super(Attack, self).step(obs)
        if obs.first():
            return self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
        
        if self.errorOccur:
            return DO_NOTHING_SC2_ACTION

        self.numStep += 1
        time.sleep(STEP_DURATION)
        sc2Action = DO_NOTHING_SC2_ACTION
        if self.num_move == 0:
            self.num_move += 1
            self.CreateState(obs)
            self.Learn()
            self.current_action = self.tables.choose_action(str(self.current_scaled_state))

            if self.current_action < ACTIONS.DO_NOTHING:
                sc2Action = DO_NOTHING_SC2_ACTION
            elif self.current_action < ACTIONS.SELECT_ALL:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    sc2Action = actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])
                    
            elif self.current_action < ACTIONS.SELECT:
                idx = self.current_action - ACTIONS.SELECT_ALL
                # if idx non valid update reward illigal move
                if self.current_scaled_state[idx + STATE.SELF_START_IDX] == 0:
                    self.currReward = REWARD_ILLIGAL_MOVE
                else:
                    startPoint = self.coordStartLocation[idx]
                    endPoint = []
                    for i in range(0,2):
                        endPoint.append(startPoint[i] + self.regionSize[i]) 

                    if SC2_Actions.SELECT_RECTANGLE in obs.observation['available_actions']:
                        sc2Action = actions.FunctionCall(SC2_Actions.SELECT_RECTANGLE, [SC2_Params.NOT_QUEUED, SwapPnt(startPoint), SwapPnt(endPoint)])

            elif self.current_action < ACTIONS.MOVE:
                idx = self.current_action - ACTIONS.SELECT
                goTo = []
                for i in range(0,2):
                    goTo.append(self.coordStartLocation[idx][i] + self.regionSize[i] / 2)

                if SC2_Actions.MOVE_IN_SCREEN in obs.observation['available_actions']:
                    sc2Action = actions.FunctionCall(SC2_Actions.MOVE_IN_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])

        else:
            self.num_move = 0
            if SC2_Actions.STOP in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])

        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0
        self.num_move = 0 
        self.currReward = 0
        self.errorOccur = False

        self.current_action = None

        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        
        self.current_scaled_state = np.zeros(STATE.SCALED_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SCALED_SIZE, dtype=np.int32, order='C')

        self.startTime = datetime.datetime.now()

        self.screenStart = [0,0]
        self.screenSize = [0,0]
        mat = (obs.observation['screen'][SC2_Params.VISIBILITY])
        pnt_y, pnt_x = mat.nonzero()
        max_y = max(pnt_y)
        min_y = min(pnt_y)

        max_x = max(pnt_x)
        min_x = min(pnt_x)

        self.screenStart[SC2_Params.Y_IDX] = min_y
        self.screenStart[SC2_Params.X_IDX] = min_x

        self.screenSize[SC2_Params.Y_IDX] = max_y - min_y + 1
        self.screenSize[SC2_Params.X_IDX] = max_x - min_x + 1

        for i in range (0,2):
            self.regionSize[i] = int((self.screenSize[i] / STATE.GRID_SIZE)) - 1

        self.coordStartLocation = []
        for y in range (0, STATE.GRID_SIZE):
            yCoord = min((y / STATE.GRID_SIZE) * self.screenSize[SC2_Params.Y_IDX], self.screenSize[SC2_Params.Y_IDX] - 1)
            for x in range (0, STATE.GRID_SIZE):
                xCoord = min((x / STATE.GRID_SIZE) * self.screenSize[SC2_Params.X_IDX], self.screenSize[SC2_Params.X_IDX] - 1)
                self.coordStartLocation.append([yCoord,xCoord])      

        # unselect army
        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
        if len(pnt_y) > 0:
            target = [pnt_x[0], pnt_y[0]]
            return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, target])
        else:
            return DO_NOTHING_SC2_ACTION

    def LastStep(self, obs):
        if obs.reward > 0:
            reward = 1
            s_ = "win"
        elif obs.reward < 0:
            reward = -1
            s_ = "loss"
        else:
            reward = -1
            s_ = "loss"

        currentDT = datetime.datetime.now()
        print("experiment terminated in " , self.numStep, "steps, duration =", currentDT - self.startTime,", in", str(currentDT))
        
        # self.CreateState(obs)
        if self.tables.end_run(str(self.previous_scaled_state), self.current_action, reward, s_):
            afterDT = datetime.datetime.now()
            print("\ttable saved!! (duration =", str(afterDT - currentDT))
        # self.currReward = reward
        # self.Learn()

    def Learn(self):
        if self.current_action is not None:
            self.tables.learn(str(self.previous_scaled_state), self.current_action, self.currReward, str(self.current_scaled_state))
            
        self.currReward = 0
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def CreateState(self, obs):
        self.CreateSelfHotZoneMat(obs)
        self.CreateEnemyHotZoneMat(obs)

        self.ScaleState(obs)


    def ScaleState(self, obs):
        self.ScaleSelfState(obs)

        for i in range (STATE.ENEMY_START_IDX, STATE.ENEMY_START_IDX + STATE.GRID_SIZE * STATE.GRID_SIZE):
            val = int(math.ceil(self.current_state[i] / NUM_UNIT_SCREEN_PIXELS))
            afterBucketing = int(math.ceil(val / STATE.POWER_BUCKETING))
            self.current_scaled_state[i] = afterBucketing * STATE.POWER_BUCKETING
        
        self.current_scaled_state[STATE.STEP_IDX] = int(self.numStep / STATE.STEP_BUCKETING)
 
    def ScaleSelfState(self, obs):

        idxSelected = self.SelectedUnitsVal()

        armySize = obs.observation['player'][SC2_Params.ARMY_SUPPLY]
        sizeEntered = 0
        for i in range (STATE.SELF_START_IDX, STATE.SELF_START_IDX + STATE.GRID_SIZE * STATE.GRID_SIZE):
            completeVal = int(math.floor(self.current_state[i] / NUM_UNIT_SCREEN_PIXELS))
            self.current_scaled_state[i] = completeVal
            self.current_state[i] -= completeVal * NUM_UNIT_SCREEN_PIXELS
            sizeEntered += completeVal

        while sizeEntered != armySize:
            maxIdx = np.argmax(self.current_state[STATE.SELF_START_IDX:STATE.ENEMY_START_IDX])
            if self.current_state[maxIdx] == 0:
                maxIdx = np.argmax(self.current_scaled_state[STATE.SELF_START_IDX:STATE.ENEMY_START_IDX])
                self.current_scaled_state[maxIdx] += armySize - sizeEntered
                break

            self.current_state[maxIdx] = 0
            self.current_scaled_state[maxIdx] += 1
            sizeEntered += 1

        for i in range (STATE.SELF_START_IDX, STATE.SELF_START_IDX + STATE.GRID_SIZE * STATE.GRID_SIZE):
            afterBucketing = int(math.ceil(self.current_scaled_state[i] / STATE.POWER_BUCKETING))
            self.current_scaled_state[i] = afterBucketing * STATE.POWER_BUCKETING

        filteredSelected = []
        occupyIdx = 0
        for i in range(0, STATE.GRID_SIZE * STATE.GRID_SIZE):
            if self.current_scaled_state[i + STATE.SELF_START_IDX] > 0: 
                if i in idxSelected:
                    filteredSelected.append(occupyIdx)
                
                occupyIdx += 1
        
        if len(filteredSelected) == 0:
            self.current_scaled_state[STATE.SELECTED_START_IDX] = STATE.SCALED_NON_SELECT
        else:
            num = 0
            for bit in filteredSelected[:]:
                num = num | (1 << bit)
            self.current_scaled_state[STATE.SELECTED_START_IDX] = num

    def SelectedUnitsVal(self):
        idxSelected = []

        for i in range(0, STATE.GRID_SIZE * STATE.GRID_SIZE):
            if self.current_state[i + STATE.SELF_START_IDX] > 0: 
                if self.current_state[i + STATE.SELECTED_START_IDX] / self.current_state[i + STATE.SELF_START_IDX] > 0.5:
                    idxSelected.append(i)
                            
        return idxSelected

    def CreateSelfHotZoneMat(self, obs):
        for i in range (STATE.SELF_START_IDX, STATE.SELF_START_IDX + STATE.GRID_SIZE * STATE.GRID_SIZE):
            self.current_state[i] = 0

        for i in range (STATE.SELECTED_START_IDX, STATE.SELECTED_START_IDX + STATE.GRID_SIZE * STATE.GRID_SIZE):
            self.current_state[i] = 0

        selectedMat = obs.observation['screen'][SC2_Params.SELECTED_IN_SCREEN]
        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
        
        for i in range(0, len(pnt_y)):

            y, x = ScaleScreenPoint(pnt_y[i], pnt_x[i], self.screenStart, self.screenSize, STATE.GRID_SIZE)
            if x > 1 or y > 1:
                self.errorOccur = True
            idx = x + y * STATE.GRID_SIZE
            self.current_state[idx + STATE.SELF_START_IDX] += 1

            if selectedMat[pnt_y[i]][pnt_x[i]]:
                self.current_state[idx + STATE.SELECTED_START_IDX] += 1

    def CreateEnemyHotZoneMat(self, obs):

        for i in range (STATE.ENEMY_START_IDX, STATE.ENEMY_START_IDX + STATE.GRID_SIZE * STATE.GRID_SIZE):
            self.current_state[i] = 0

        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()
        for i in range(0, len(pnt_y)):
            y, x = ScaleScreenPoint(pnt_y[i], pnt_x[i], self.screenStart, self.screenSize, STATE.GRID_SIZE)
            if x > 1 or y > 1:
                self.errorOccur = True

            idx = x + y * STATE.GRID_SIZE
            self.current_state[idx + STATE.ENEMY_START_IDX] += 1

    def PrintState(self):
        print("self mat :")
        for y in range(0, STATE.GRID_SIZE):
            for x in range(0, STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[idx], end = ' ')
            print('|')

        print("\nenemy mat :")
        for y in range(0, STATE.GRID_SIZE):
            for x in range(0, STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE + STATE.GRID_SIZE * STATE.GRID_SIZE
                print(self.current_scaled_state[idx], end = ' ')
            print('|')

        


