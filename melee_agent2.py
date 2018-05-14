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
Q_TABLE_SMART_EXPLORATION = "melee_attack_2_qtable_smartExploration"
T_TABLE_SMART_EXPLORATION = "melee_attack_2_ttable_smartExploration"
R_TABLE_SMART_EXPLORATION = "melee_attack_2_rtable_smartExploration"
RESULT_SMART_EXPLORATION = "melee_attack_2_result_smartExploration"

Q_TABLE_NAIVE_EXPLORATION = "melee_attack_2_qtable_naiveExploration"
T_TABLE_NAIVE_EXPLORATION = "melee_attack_2_ttable_naiveExploration"
R_TABLE_NAIVE_EXPLORATION = "melee_attack_2_rtable_naiveExploration"
RESULT_NAIVE_EXPLORATION = "melee_attack_2_result_naiveExploration"

GRID_SIZE = 4

NON_VALID_NUM = -1

DO_NOTHING_ACTION = 1
SELECT_ALL_ACTION = DO_NOTHING_ACTION + 1
SELECT_ACTIONS = GRID_SIZE * GRID_SIZE + SELECT_ALL_ACTION
MOVE_ACTIONS = GRID_SIZE * GRID_SIZE + SELECT_ACTIONS
NUM_ACTIONS = MOVE_ACTIONS

# non-scaled state details
STATE_SIZE = 3 * GRID_SIZE * GRID_SIZE
SCALED_STATE_SIZE = 2 * GRID_SIZE * GRID_SIZE + 1

STATE_SELF_START_IDX = 0
STATE_ENEMY_START_IDX = GRID_SIZE * GRID_SIZE
STATE_SELECTED_START_IDX = 2 * GRID_SIZE * GRID_SIZE

SCALED_STATE_NON_SELECT = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

REWARD_ILLIGAL_MOVE = -1

STEP_DURATION = 0

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


def ScaleScreenPoint(y, x, newGridSize):
    y /= SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX]
    x /= SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]

    y *= newGridSize
    x *= newGridSize    

    return int(y), int(x)

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

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        
        self.current_scaled_state = np.zeros(SCALED_STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(SCALED_STATE_SIZE, dtype=np.int32, order='C')

        # model params
        xScreenSize = SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]
        yScreenSize = SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX]
        self.regionSize = [yScreenSize, xScreenSize]
        for i in range (0,2):
            self.regionSize[i] = int((self.regionSize[i] / GRID_SIZE)) - 1

        self.coordStartLocation = []

        for y in range (0, GRID_SIZE):
            yCoord = min((y / GRID_SIZE) * yScreenSize, yScreenSize - 1)
            for x in range (0, GRID_SIZE):
                xCoord = min((x / GRID_SIZE) * xScreenSize, xScreenSize - 1)
                self.coordStartLocation.append([yCoord,xCoord])        

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
            self.current_action = self.tables.choose_action(str(self.current_state))
            if self.current_action < DO_NOTHING_ACTION:
                sc2Action = DO_NOTHING_SC2_ACTION
            elif self.current_action < SELECT_ALL_ACTION:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    sc2Action = actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])
                    
            elif self.current_action < SELECT_ACTIONS:
                idx = self.current_action - SELECT_ALL_ACTION
                # if idx non valid update reward illigal move
                if self.current_scaled_state[idx + STATE_SELF_START_IDX] == 0:
                    self.currReward = REWARD_ILLIGAL_MOVE
                else:
                    startPoint = self.coordStartLocation[idx]
                    endPoint = []
                    for i in range(0,2):
                        endPoint.append(startPoint[i] + self.regionSize[i]) 

                    if SC2_Actions.SELECT_RECTANGLE in obs.observation['available_actions']:
                        sc2Action = actions.FunctionCall(SC2_Actions.SELECT_RECTANGLE, [SC2_Params.NOT_QUEUED, SwapPnt(startPoint), SwapPnt(endPoint)])

            elif self.current_action < MOVE_ACTIONS:
                idx = self.current_action - SELECT_ACTIONS
                goTo = self.coordStartLocation[idx]
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
        elif obs.reward < 0:
            reward = -1
        else:
            reward = -1

        self.tables.end_run(str(self.previous_scaled_state), self.current_action, reward)

    def Learn(self):
        if self.current_action is not None:
            self.tables.learn(str(self.previous_scaled_state), self.current_action, self.currReward, str(self.current_scaled_state))
            # if self.current_action < SELECT_ACTIONS and self.current_action >= DO_NOTHING_ACTION and self.currReward == 0:
            #     if self.current_action < SELECT_ALL_ACTION:
            #         print("\nselect all, prev select val =" , self.previous_scaled_state[STATE_SELECTED_START_IDX], "current select val =", self.current_scaled_state[STATE_SELECTED_START_IDX])
            #     else:
            #         print("prev select val =" , self.previous_scaled_state[STATE_SELECTED_START_IDX], "current select val =", self.current_scaled_state[STATE_SELECTED_START_IDX])
                
            #     num = self.current_scaled_state[STATE_SELECTED_START_IDX]
            #     for i in range (STATE_SELF_START_IDX, STATE_SELF_START_IDX + GRID_SIZE * GRID_SIZE):
            #         if self.current_scaled_state[i] > 0:
            #             num = num >> 1
            #     if num > 0:
            #         print ("Error in num")
            #         time.sleep(5)


        self.currReward = 0
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def CreateState(self, obs):
        self.CreateSelfHotZoneMat(obs)
        self.CreateEnemyHotZoneMat(obs)

        self.ScaleState(obs)

    def ScaleState(self, obs):
        self.ScaleSelfState(obs)

        for i in range (STATE_ENEMY_START_IDX, STATE_ENEMY_START_IDX + GRID_SIZE * GRID_SIZE):
            self.current_scaled_state[i] = int(math.ceil(self.current_state[i] / TerranUnit.MARINE_SCREEN_NUM_PIXELS))
    
    def ScaleSelfState(self, obs):

        idxSelected = self.SelectedUnitsVal()

        armySize = obs.observation['player'][SC2_Params.ARMY_SUPPLY]
        sizeEntered = 0
        for i in range (STATE_SELF_START_IDX, STATE_SELF_START_IDX + GRID_SIZE * GRID_SIZE):
            completeVal = int(math.floor(self.current_state[i] / TerranUnit.MARINE_SCREEN_NUM_PIXELS))
            self.current_scaled_state[i] = completeVal
            self.current_state[i] -= completeVal * TerranUnit.MARINE_SCREEN_NUM_PIXELS
            sizeEntered += completeVal

        while sizeEntered != armySize:
            maxIdx = np.argmax(self.current_state[STATE_SELF_START_IDX:STATE_ENEMY_START_IDX])
            if self.current_state[maxIdx] == 0:
                maxIdx = np.argmax(self.current_scaled_state[STATE_SELF_START_IDX:STATE_ENEMY_START_IDX])
                self.current_scaled_state[maxIdx] += armySize - sizeEntered
                break

            self.current_state[maxIdx] = 0
            self.current_scaled_state[maxIdx] += 1
            sizeEntered += 1

        filteredSelected = []
        occupyIdx = 0
        for i in range(0, GRID_SIZE * GRID_SIZE):
            if self.current_scaled_state[i + STATE_SELF_START_IDX] > 0: 
                if i in idxSelected:
                    filteredSelected.append(occupyIdx)
                
                occupyIdx += 1
        
        if len(filteredSelected) == 0:
            self.current_scaled_state[STATE_SELECTED_START_IDX] = SCALED_STATE_NON_SELECT
        else:
            num = 0
            for bit in filteredSelected[:]:
                num = num | (1 << bit)
            self.current_scaled_state[STATE_SELECTED_START_IDX] = num

    def SelectedUnitsVal(self):
        idxSelected = []

        for i in range(0, GRID_SIZE * GRID_SIZE):
            if self.current_state[i + STATE_SELF_START_IDX] > 0: 
                if self.current_state[i + STATE_SELECTED_START_IDX] / self.current_state[i + STATE_SELF_START_IDX] > 0.5:
                    idxSelected.append(i)
                            
        return idxSelected

    def CreateSelfHotZoneMat(self, obs):
        for i in range (STATE_SELF_START_IDX, STATE_SELF_START_IDX + GRID_SIZE * GRID_SIZE):
            self.current_state[i] = 0

        for i in range (STATE_SELECTED_START_IDX, STATE_SELECTED_START_IDX + GRID_SIZE * GRID_SIZE):
            self.current_state[i] = 0

        selectedMat = obs.observation['screen'][SC2_Params.SELECTED_IN_SCREEN]
        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
        
        for i in range(0, len(pnt_y)):
            y, x = ScaleScreenPoint(pnt_y[i], pnt_x[i], GRID_SIZE)
            idx = x + y * GRID_SIZE
            self.current_state[idx + STATE_SELF_START_IDX] += 1

            if selectedMat[pnt_y[i]][pnt_x[i]]:
                self.current_state[idx + STATE_SELECTED_START_IDX] += 1

    def CreateEnemyHotZoneMat(self, obs):

        for i in range (STATE_ENEMY_START_IDX, STATE_ENEMY_START_IDX + GRID_SIZE * GRID_SIZE):
            self.current_state[i] = 0

        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()
        for i in range(0, len(pnt_y)):
            y, x = ScaleScreenPoint(pnt_y[i], pnt_x[i], GRID_SIZE)
            idx = x + y * GRID_SIZE
            self.current_state[idx + STATE_ENEMY_START_IDX] += 1

    def PrintState(self):
        print("self mat :")
        for y in range(0, GRID_SIZE):
            for x in range(0, GRID_SIZE):
                idx = x + y * GRID_SIZE
                print(self.current_state[idx], end = ' ')
            print('|')

        print("\nenemy mat :")
        for y in range(0, GRID_SIZE):
            for x in range(0, GRID_SIZE):
                idx = x + y * GRID_SIZE + GRID_SIZE * GRID_SIZE
                print(self.current_state[idx], end = ' ')
            print('|')


