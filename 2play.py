
import numpy as np
import time
import math
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions
from utils import QLearningTable

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintMiniMap
from utils import PrintSpecificMat
from utils import FindMiddle

STEP_DURATION = 0.1

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

def ScaleScreenPoint(y, x, newGridSize):
    y /= SC2_Params.SCREEN_SIZE[SC2_Params.Y_IDX]
    x /= SC2_Params.SCREEN_SIZE[SC2_Params.X_IDX]

    y *= newGridSize
    x *= newGridSize    

    return int(y), int(x)

def GetSelfLocation(obs):
    self_y, self_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
    if len(self_y) > 0:
        return FindMiddle(self_y, self_x)
    else:
        return [-1,-1]

class Play(base_agent.BaseAgent):
    def __init__(self):        
        super(Play, self).__init__()

        # states and action:
        self.current_action = None

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

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

        self.move_action = 0   

    def step(self, obs):
        super(Play, self).step(obs)
        if obs.first():
            self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
        
        if self.errorOccur:
            return DO_NOTHING_SC2_ACTION

        sc2Action = DO_NOTHING_SC2_ACTION
        time.sleep(STEP_DURATION)
        # self.numStep += 1
        # time.sleep(STEP_DURATION)
        # sc2Action = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])
        
        # if self.move_action == 0:
        #     self.move_action += 1
        #     self.CreateState(obs)
        #     a = input("enter action ")
        #     self.current_action = int (a)
        #     # self.PrintState()

        #     if self.current_action < DO_NOTHING_ACTION:
        #         sc2Action = DO_NOTHING_SC2_ACTION
        #     elif self.current_action < SELECT_ALL_ACTION:
        #         if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
        #             sc2Action = actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])
                    
        #     elif self.current_action < SELECT_ACTIONS:
        #         idx = self.current_action - SELECT_ALL_ACTION
        #         startPoint = self.coordStartLocation[idx]
        #         endPoint = []
        #         for i in range(0,2):
        #             endPoint.append(startPoint[i] + self.regionSize[i]) 

        #         if SC2_Actions.SELECT_RECTANGLE in obs.observation['available_actions']:
        #             sc2Action = actions.FunctionCall(SC2_Actions.SELECT_RECTANGLE, [SC2_Params.NOT_QUEUED, SwapPnt(startPoint), SwapPnt(endPoint)])

        #     elif self.current_action < MOVE_ACTIONS:
        #         idx = self.current_action - SELECT_ACTIONS
        #         goTo = []
        #         for i in range(0,2):
        #             goTo.append(self.coordStartLocation[idx][i] + self.regionSize[i] / 2)
        #         if SC2_Actions.MOVE_IN_SCREEN in obs.observation['available_actions']:
        #             sc2Action = actions.FunctionCall(SC2_Actions.MOVE_IN_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
        
        # elif self.move_action == 1:
        #     self.move_action = 0
        #     if SC2_Actions.STOP in obs.observation['available_actions']:
        #         sc2Action = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])

        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0
             
        self.errorOccur = False

    def LastStep(self, obs):
        if obs.reward > 0:
            print("won")
            reward = 1
        elif obs.reward < 0:
            reward = -1
            print("loss")
        else:
            reward = -1
            print("loss")

    def CreateState(self, obs):
        self.CreateSelfHotZoneMat(obs)
        self.CreateEnemyHotZoneMat(obs)

        self.ScaleState(obs)
        
        if self.current_action is not None and self.current_action >= SELECT_ACTIONS:
            selected = self.previous_scaled_state[STATE_SELECTED_START_IDX]
            idx = 0
            if selected > 0:
                for y in range(0, GRID_SIZE):
                    for x in range(0, GRID_SIZE):
                        idx = x + y * GRID_SIZE
                        print (self.previous_scaled_state[idx + STATE_SELF_START_IDX], end = ' ')

                    print ("  -->  ", end = '')
                    for x in range(0, GRID_SIZE):
                        idx = x + y * GRID_SIZE
                        print (self.current_scaled_state[idx + STATE_SELF_START_IDX], end = ' ')

                    print ("|")
                print ("selected units =", selected, " go to location =", self.current_action - SELECT_ACTIONS)
        self.previous_scaled_state[:] = self.current_scaled_state[:]

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