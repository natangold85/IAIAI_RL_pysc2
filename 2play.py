
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

# state details
STATE_SIZE = 2 * GRID_SIZE * GRID_SIZE
STATE_ENEMY_START = GRID_SIZE * GRID_SIZE

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

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

        self.numStep += 1
        time.sleep(STEP_DURATION)
        sc2Action = DO_NOTHING_SC2_ACTION
        sc2Action = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])
        
        if self.move_action == 0:
            self.move_action += 1
            self.CreateState(obs)
            a = input("enter action ")
            self.current_action = int (a)
            # self.PrintState()

            if self.current_action < DO_NOTHING_ACTION:
                sc2Action = DO_NOTHING_SC2_ACTION
            elif self.current_action < SELECT_ALL_ACTION:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    sc2Action = actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])
                    
            elif self.current_action < SELECT_ACTIONS:
                idx = self.current_action - SELECT_ALL_ACTION
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
        
        elif self.move_action == 1:
            self.move_action = 0
            if SC2_Actions.STOP in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])

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
        armySize = obs.observation['player'][SC2_Params.ARMY_SUPPLY]
        self.CreateSelfHotZoneMat(obs, SC2_Params.PLAYER_SELF , armySize)
        self.CreateEnemyHotZoneMat(obs, SC2_Params.PLAYER_HOSTILE)
        
        # selfIdx = 0
        # enemyIdx = GRID_SIZE * GRID_SIZE
        # for y in range(0, GRID_SIZE):
        #     for x in range(0, GRID_SIZE):
        #         print(self.current_state[selfIdx], end = ' ')
        #         selfIdx += 1

        #     print ("|    ", end = " ")
        #     for x in range(0, GRID_SIZE):
        #         print(self.current_state[enemyIdx], end = ' ')
        #         enemyIdx += 1


        #     print ("|")

    def CreateSelfHotZoneMat(self, obs, playerType, armySize):
        for i in range (0, GRID_SIZE * GRID_SIZE):
            self.current_state[i] = 0


        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == playerType).nonzero()
        for i in range(0, len(pnt_y)):
            y, x = ScaleScreenPoint(pnt_y[i], pnt_x[i], GRID_SIZE)
            idx = x + y * GRID_SIZE
            self.current_state[idx] += 1
        
        for i in range (0, GRID_SIZE * GRID_SIZE):
            self.current_state[i] = int(math.ceil(self.current_state[i] / TerranUnit.MARINE_SCREEN_NUM_PIXELS))

    def CreateEnemyHotZoneMat(self, obs, playerType):
        for i in range (STATE_ENEMY_START, STATE_ENEMY_START + GRID_SIZE * GRID_SIZE):
            self.current_state[i] = 0

        pnt_y, pnt_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == playerType).nonzero()
        for i in range(0, len(pnt_y)):
            y, x = ScaleScreenPoint(pnt_y[i], pnt_x[i], GRID_SIZE)
            idx = STATE_ENEMY_START + x + y * GRID_SIZE
            self.current_state[idx] += 1
        
        for i in range (STATE_ENEMY_START, STATE_ENEMY_START + GRID_SIZE * GRID_SIZE):
            self.current_state[i] = int(math.ceil(self.current_state[i] / TerranUnit.MARINE_SCREEN_NUM_PIXELS))

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