import random
import math
import os.path
import logging
import traceback

#udp
import socket
import threading

import numpy as np
import pandas as pd
import time
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

# ACTIONS

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

# build actions
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_BUILD_OIL_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id

# building additions
_BUILD_REACTOR = actions.FUNCTIONS.Build_Reactor_screen.id
_BUILD_TECHLAB = actions.FUNCTIONS.Build_TechLab_screen.id

# train army action
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

_TRAIN_HELLION = actions.FUNCTIONS.Train_Hellion_quick.id
_TRAIN_SIEGE_TANK = actions.FUNCTIONS.Train_SiegeTank_quick.id


_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_CAMERA = features.MINIMAP_FEATURES.camera.index
_HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

#player general info
_MINERALS = 1
_VESPENE = 2
_SUPPLY_CAP = 4
_IDLE_WORKER_COUNT = 7

# multi and single select information
_BUILDING_COMPLETION_IDX = 6

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_OIL_REFINERY = 20
_TERRAN_BARRACKS = 21
_TERRAN_FACTORY = 27
_TERRAN_REACTOR = 38
_TERRAN_TECHLAB = 39

_TERRAN_ARMY= [53,40,49,33]

_TERRAN_FLYING_BARRACKS = 46
_TERRAN_FLYING_FACTORY = 43

_NEUTRAL_MINERAL_FIELD = [341, 483]
_VESPENE_GAS_FIELD = [342]

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_SINGLE = [0]
_SELECT_ALL = [2]

BUILDING_NAMES = {}
BUILDING_NAMES[_TERRAN_COMMANDCENTER] = "CommandCenter"
BUILDING_NAMES[_TERRAN_SUPPLY_DEPOT] = "SupplyDepot"
BUILDING_NAMES[_TERRAN_BARRACKS] = "Barracks"
BUILDING_NAMES[_TERRAN_FACTORY] = "Factory"
BUILDING_NAMES[_TERRAN_OIL_REFINERY] = "OilRefinery"

BUILDING_NAMES[_TERRAN_REACTOR] = "Reactor"
BUILDING_NAMES[_TERRAN_TECHLAB] = "TechLab"

BUILDING_SIZES = {}
BUILDING_SIZES[_TERRAN_COMMANDCENTER] = 18
BUILDING_SIZES[_TERRAN_SUPPLY_DEPOT] = 9
BUILDING_SIZES[_TERRAN_BARRACKS] = 12
BUILDING_SIZES[_TERRAN_FACTORY] = 12

BUILDING_SIZES[_TERRAN_REACTOR] = 3
BUILDING_SIZES[_TERRAN_TECHLAB] = 3

BUIILDING_2_SC2ACTIONS = {}
BUIILDING_2_SC2ACTIONS[_TERRAN_OIL_REFINERY] = _BUILD_OIL_REFINERY
BUIILDING_2_SC2ACTIONS[_TERRAN_SUPPLY_DEPOT] = _BUILD_SUPPLY_DEPOT
BUIILDING_2_SC2ACTIONS[_TERRAN_BARRACKS] = _BUILD_BARRACKS
BUIILDING_2_SC2ACTIONS[_TERRAN_FACTORY] = _BUILD_FACTORY

UNIT_CHAR = {}
UNIT_CHAR[_TERRAN_COMMANDCENTER] = 'C'
UNIT_CHAR[_TERRAN_SCV] = 's'
UNIT_CHAR[_TERRAN_SUPPLY_DEPOT] = 'S'
UNIT_CHAR[_TERRAN_OIL_REFINERY] = 'G'
UNIT_CHAR[_TERRAN_BARRACKS] = 'B'
UNIT_CHAR[_TERRAN_FACTORY] = 'F'
UNIT_CHAR[_TERRAN_REACTOR] = 'R'
UNIT_CHAR[_TERRAN_TECHLAB] = 'T'

UNIT_CHAR[_TERRAN_FLYING_BARRACKS] = 'Y'
UNIT_CHAR[_TERRAN_FLYING_FACTORY] = 'Y'

DO_NOTHING_BUILDING_CHECK = [_TERRAN_COMMANDCENTER, _TERRAN_SUPPLY_DEPOT, _TERRAN_OIL_REFINERY, _TERRAN_BARRACKS, _TERRAN_FACTORY]
for field in _NEUTRAL_MINERAL_FIELD[:]:
    UNIT_CHAR[field] = 'm'
for gas in _VESPENE_GAS_FIELD[:]:
    UNIT_CHAR[gas] = 'g'
for army in _TERRAN_ARMY[:]:
    UNIT_CHAR[army] = 'a'

SUPPLY_SIZE_CC = 15
SUPPLY_SIZE_SD = 8

Q_TABLE_FILE = 'buildbase_q_table'

ID_ACTION_DO_NOTHING = 0
ID_ACTION_BUILD_SUPPLY_DEPOT = 1
ID_ACTION_BUILD_BARRACKS = 2
ID_ACTION_BUILD_REFINERY = 3
ID_ACTION_BUILD_FACTORY = 4
ID_ACTION_TRAIN_ARMY = 5

NUM_ACTIONS = 6

BUILDING_2_ACTION_TRANSITION = {}
BUILDING_2_ACTION_TRANSITION[_TERRAN_SUPPLY_DEPOT] = ID_ACTION_BUILD_SUPPLY_DEPOT
BUILDING_2_ACTION_TRANSITION[_TERRAN_OIL_REFINERY] = ID_ACTION_BUILD_REFINERY
BUILDING_2_ACTION_TRANSITION[_TERRAN_BARRACKS] = ID_ACTION_BUILD_BARRACKS
BUILDING_2_ACTION_TRANSITION[_TERRAN_FACTORY] = ID_ACTION_BUILD_FACTORY

ACTION_2_BUILDING_TRANSITION = {}
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_SUPPLY_DEPOT] = _TERRAN_SUPPLY_DEPOT
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_REFINERY] = _TERRAN_OIL_REFINERY
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_BARRACKS] = _TERRAN_BARRACKS
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_FACTORY] = _TERRAN_FACTORY


DONOTHING_ACTION_NOTHING = 0
DONOTHING_ACTION_LAND_FACTORY = 1
DONOTHING_ACTION_LAND_BARRACKS = 2
DONOTHING_ACTION_IDLE_WORKER = 3
DONOTHING_ACTION_PRINT_QUEUE = 4
Y_IDX = 0
X_IDX = 1

MINIMAP_SIZE = [64, 64]
MAX_MINIMAP_DIST = MINIMAP_SIZE[X_IDX] * MINIMAP_SIZE[X_IDX] + MINIMAP_SIZE[Y_IDX] * MINIMAP_SIZE[Y_IDX] 
MINI_MAP_START = [8, 8]
MINI_MAP_END = [61, 50]

SCREEN_SIZE = [84, 84]

STEP_DURATION = 0

# state details
STATE_NON_VALID_NUM = -1

STATE_MINERALS_MAX = 500
STATE_GAS_MAX = 300
STATE_MINERALS_BUCKETING = 50
STATE_GAS_BUCKETING = 50

STATE_COMMAND_CENTER_IDX = 0
STATE_MINERALS_IDX = 1
STATE_GAS_IDX = 2
STATE_SUPPLY_DEPOT_IDX = 3
STATE_REFINERY_IDX = 4
STATE_BARRACKS_IDX = 5
STATE_FACTORY_IDX = 6
STATE_IN_PROGRESS_SUPPLY_DEPOT_IDX = 7
STATE_IN_PROGRESS_REFINERY_IDX = 8
STATE_IN_PROGRESS_BARRACKS_IDX = 9
STATE_IN_PROGRESS_FACTORY_IDX = 10

STATE_SIZE = 11

BUILDING_2_STATE_TRANSITION = {}
BUILDING_2_STATE_TRANSITION[_TERRAN_COMMANDCENTER] = [STATE_COMMAND_CENTER_IDX, -1]
BUILDING_2_STATE_TRANSITION[_TERRAN_SUPPLY_DEPOT] = [STATE_SUPPLY_DEPOT_IDX, STATE_IN_PROGRESS_SUPPLY_DEPOT_IDX]
BUILDING_2_STATE_TRANSITION[_TERRAN_OIL_REFINERY] = [STATE_REFINERY_IDX, STATE_IN_PROGRESS_REFINERY_IDX]
BUILDING_2_STATE_TRANSITION[_TERRAN_BARRACKS] = [STATE_BARRACKS_IDX, STATE_IN_PROGRESS_BARRACKS_IDX]
BUILDING_2_STATE_TRANSITION[_TERRAN_FACTORY] = [STATE_FACTORY_IDX, STATE_IN_PROGRESS_FACTORY_IDX]


REWARD_TRAIN_MARINE = 1
REWARD_TRAIN_REAPER = 2
REWARD_TRAIN_HELLION = 4
REWARD_TRAIN_SIEGE_TANK = 6
NORMALIZED_REWARD = 300

CREATE_DETAILED_STATE = False
DETAILED_STATE_SIZE = 6

class BuildingCoord:
    def __init__(self, screenLocation, isAddition = False):
        self.m_screenLocation = screenLocation
        self.m_buildingAddition = isAddition

class BuildingCmd:
    def __init__(self, screenLocation, inProgress = False):
        self.m_screenLocation = screenLocation
        self.m_inProgress = inProgress
        self.m_steps2Check = 20


class AttackCmd:
    def __init__(self, state, attackCoord, isBaseAttack ,attackStarted = False):
        self.m_state = state
        self.m_attackCoord = [math.ceil(attackCoord[Y_IDX]), math.ceil(attackCoord[X_IDX])]
        self.m_inTheWayBattle = [-1, -1]
        self.m_isBaseAttack = isBaseAttack
        self.m_attackStarted = attackStarted
        self.m_attackEnded = False  



# utils function
def Min(points):
    minVal = points[0]
    for i in range(1, len(points)):
        minVal = min(minVal, points[i])

    return minVal

def Max(points):
    maxVal = points[0]
    for i in range(1, len(points)):
        maxVal = max(maxVal, points[i])

    return maxVal

def FindMiddle(points_y, points_x):
    min_x = Min(points_x)
    max_x = Max(points_x)
    midd_x = min_x + (max_x - min_x) / 2

    min_y = Min(points_y)
    max_y = Max(points_y)
    midd_y = min_y + (max_y - min_y) / 2

    return [int(midd_y), int(midd_x)]

def IsInScreen(y,x):
    return y >= 0 and y < SCREEN_SIZE[Y_IDX] and x >= 0 and x < SCREEN_SIZE[X_IDX] 

def Flood(location, buildingMap):   
    closeLocs = [[location[Y_IDX] + 1, location[X_IDX]], [location[Y_IDX] - 1, location[X_IDX]], [location[Y_IDX], location[X_IDX] + 1], [location[Y_IDX], location[X_IDX] - 1] ]
    points_y = [location[Y_IDX]]
    points_x = [location[X_IDX]]
    for loc in closeLocs[:]:
        if IsInScreen(loc[Y_IDX],loc[X_IDX]) and buildingMap[loc[Y_IDX]][loc[X_IDX]]:
            buildingMap[loc[Y_IDX]][loc[X_IDX]] = False
            pnts_y, pnts_x = Flood(loc, buildingMap)
            points_x.extend(pnts_x)
            points_y.extend(pnts_y)  

    return points_y, points_x


def IsolateArea(location, buildingMap):           
    return Flood(location, buildingMap)

def Scale2MiniMap(point, camNorthWestCorner, camSouthEastCorner):
    scaledPoint = [0,0]
    scaledPoint[Y_IDX] = point[Y_IDX] * (camSouthEastCorner[Y_IDX] - camNorthWestCorner[Y_IDX]) / SCREEN_SIZE[Y_IDX] 
    scaledPoint[X_IDX] = point[X_IDX] * (camSouthEastCorner[X_IDX] - camNorthWestCorner[X_IDX]) / SCREEN_SIZE[X_IDX] 
    
    scaledPoint[Y_IDX] += camNorthWestCorner[Y_IDX]
    scaledPoint[X_IDX] += camNorthWestCorner[X_IDX]
    
    scaledPoint[Y_IDX] = math.ceil(scaledPoint[Y_IDX])
    scaledPoint[X_IDX] = math.ceil(scaledPoint[X_IDX])

    return scaledPoint

def Scale2Screen(point, camNorthWestCorner, camSouthEastCorner):
    scaledPoint = [0,0]
    scaledPoint[Y_IDX] = point[Y_IDX] - camNorthWestCorner[Y_IDX]
    scaledPoint[X_IDX] = point[X_IDX] - camNorthWestCorner[X_IDX]

    scaledPoint[Y_IDX] = int(scaledPoint[Y_IDX] * SCREEN_SIZE[Y_IDX] / (camSouthEastCorner[Y_IDX] - camNorthWestCorner[Y_IDX]))
    scaledPoint[X_IDX] = int(scaledPoint[X_IDX] * SCREEN_SIZE[X_IDX] /  (camSouthEastCorner[X_IDX] - camNorthWestCorner[X_IDX]))

    return scaledPoint

def GetScreenCorners(obs):
    cameraLoc = obs.observation['minimap'][_CAMERA]
    ca_y, ca_x = cameraLoc.nonzero()

    return [ca_y.min(), ca_x.min()] , [ca_y.max(), ca_x.max()]

def PowerSurroundPnt(point, radius2Include, powerMat):
    power = 0
    for y in range(-radius2Include, radius2Include):
        for x in range(-radius2Include, radius2Include):
            power += powerMat[y + point[Y_IDX]][x + point[X_IDX]]

    return power
def BattleStarted(selfMat, enemyMat):
    attackRange = 1

    for xEnemy in range (attackRange, MINIMAP_SIZE[X_IDX] - attackRange):
        for yEnemy in range (attackRange, MINIMAP_SIZE[Y_IDX] - attackRange):
            if enemyMat[yEnemy,xEnemy]:
                for xSelf in range(xEnemy - attackRange, xEnemy + attackRange):
                    for ySelf in range(yEnemy - attackRange, yEnemy + attackRange):
                        if enemyMat[ySelf][xSelf]:
                            return True, yEnemy, xEnemy

    return False, -1, -1

def PrintSpecificMat(mat, range2Include = 0):
    for y in range(range2Include, MINIMAP_SIZE[Y_IDX] - range2Include):
        for x in range(range2Include, MINIMAP_SIZE[X_IDX] - range2Include):
            if range2Include == 0:
                if mat[y][x]:
                    print ('1', end = '')
                else:
                    print ('0', end = '')
            else:
                sPower = PowerSurroundPnt([y,x], range2Include, mat)
                if sPower > 9:
                    print(sPower, end = ' ')
                else:
                    print(sPower, end = '  ')
        print('|')

def PrintSpecificMatAndPnt(mat, points = [], range2Include = 0):
    
    for y in range(range2Include, MINIMAP_SIZE[Y_IDX] - range2Include):
        for x in range(range2Include, MINIMAP_SIZE[X_IDX] - range2Include):           
            if range2Include == 0:
                if mat[y][x]:
                    print ('1', end = '')
                else:
                    prnted = False
                    for i in range(0, len(points)):
                        if x == points[i][X_IDX] and y == points[i][Y_IDX]:
                            print("P", end = '')
                            prnted = True
                            break
                    if not prnted:
                        print ('0', end = '')
            else:
                prnted = False
                for i in range(0, len(points)):
                    if x == points[i][X_IDX] and y == points[i][Y_IDX]:
                        print("P", end = '')
                        prnted = True
                        break
                if not prnted:
                    sPower = PowerSurroundPnt([y,x], range2Include, mat)
                    if sPower > 9:
                        print(sPower, end = ' ')
                    else:
                        print(sPower, end = '  ')
        print('|')

def SwapPnt(point):
    return point[1], point[0]

def GetCoord(idxLocation, gridSize_x):
    ret = [-1,-1]
    ret[Y_IDX] = int(idxLocation / gridSize_x)
    ret[X_IDX] = idxLocation % gridSize_x
    return ret

def IsBuildAction(action):
    return action >= ID_ACTION_BUILD_SUPPLY_DEPOT and action < ID_ACTION_TRAIN_ARMY

def PrintBuildingSizes(unit_type):
    ccMap = unit_type == _TERRAN_COMMANDCENTER
    PrintSingleBuildingSize(ccMap, "command center")
    sdMap = unit_type == _TERRAN_SUPPLY_DEPOT
    PrintSingleBuildingSize(sdMap, "supply depot")
    baMap = unit_type == _TERRAN_BARRACKS
    PrintSingleBuildingSize(baMap, "barracks")

def PrintSingleBuildingSize(buildingMap, name):
    allPnts_y, allPnts_x = buildingMap.nonzero()
    if len(allPnts_y > 0):
        pnts_y, pnts_x = IsolateArea([allPnts_y[0], allPnts_x[0]], buildingMap)
        size_y = Max(pnts_y) - Min(pnts_y)
        size_x = Max(pnts_x) - Min(pnts_x)
        print(name , "size x = ", size_x, "size y = ", size_y)

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float)
        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        self.NumRunsExperimentSlot = 1
        
        self.AvgRewardSlot = 2
        self.AvgRewardExperimentSlot = 3

        self.check_state_exist(self.TrialsData)
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 0
        self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 0

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s)
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            self.check_state_exist(s_)
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
    def end_run(self, r):
        
        numTotalRuns = self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot]
        numExpRuns = self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot]
        
        totReward = self.q_table.ix[self.TrialsData, self.AvgRewardSlot]
        expReward = self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot]

        totReward = (numTotalRuns * totReward + r) / (numTotalRuns + 1)
        expReward = (numExpRuns * expReward + r) / (numExpRuns + 1)

        self.q_table.ix[self.TrialsData, self.AvgRewardSlot] = totReward
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = expReward

        
        self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot] = numTotalRuns + 1
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = numExpRuns + 1

        print("num total runs = ", numTotalRuns, "avg total = ", totReward)
        print("num experiment runs = ", numExpRuns, "avg experiment = ", expReward)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class BuildBase(base_agent.BaseAgent):
    def __init__(self):
        super(BuildBase, self).__init__()
        
        # qtables:
        self.qTable = QLearningTable(actions=list(range(NUM_ACTIONS)))
        if os.path.isfile(Q_TABLE_FILE + '.gz'):
            self.qTable.q_table = pd.read_pickle(Q_TABLE_FILE + '.gz', compression='gzip')

        # states and action:
        self.previous_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        if CREATE_DETAILED_STATE:
            self.previous_Detailedstate = np.zeros(DETAILED_STATE_SIZE, dtype=np.int32, order='C')
            self.current_Detailedstate = np.zeros(DETAILED_STATE_SIZE, dtype=np.int32, order='C')

        # decision maker
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.current_state_for_decision_making = None

        # model params
        self.unit_type = None

        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]
        
        self.topLeftBaseLocation = [23,18]
        self.bottomRightBaseLocation = [45,39]

        self.move_number = 0
        self.step_num = 0
        self.doNothingAction = DONOTHING_ACTION_NOTHING

        self.buildingVec = {}
        self.buildCommands = {}

        self.actionSucceed = True
        self.IsMultiSelect = False
        self.donothingBuildingIdx = 0
        self.currentBuildingTypeSelected = 0
               
        self.nextTrainArmyBuilding = _TERRAN_BARRACKS
        self.lastBuildingIdx = -1

        self.lastTrainArmyReward = 0
        self.accumulatedReward = 0
        
        # for developing:
        self.bool = False
        self.coord = [-1,-1]

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def transformLocation(self, y, x):
        if not self.base_top_left:
            return [64 - y, 64 - x]
        
        return [y, x]

    def step(self, obs):
        super(BuildBase, self).step(obs)
        self.step_num += 1
        
        try:
            self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
            self.unit_type = obs.observation['screen'][_UNIT_TYPE]
            
            if obs.last():
                self.LastStep(obs)
                return actions.FunctionCall(_NO_OP, [])
            elif obs.first():
                self.FirstStep(obs)
            
            time.sleep(STEP_DURATION)
            self.UpdateBuildingInProgress(obs)
            if self.IsMultiSelect:
                self.UpdateBuildingCompletion(obs, self.currentBuildingTypeSelected)
                self.IsMultiSelect = False

            if self.move_number == 0:
                self.CreateState(obs)
                
                self.move_number += 1
                action = self.qTable.choose_action(str(self.current_scaled_state))

                if self.previous_action is not None:
                    self.Insert2QTable()

                self.previous_state[:] = self.current_state[:]
                self.previous_scaled_state[:] = self.current_scaled_state[:]
                self.previous_action = action
                self.actionSucceed = True

                if action == ID_ACTION_DO_NOTHING:
                    # search for flying building
                    flyingBa_y, flyingBa_x = (self.unit_type == _TERRAN_FLYING_BARRACKS).nonzero()
                    flyingFa_y, flyingFa_x = (self.unit_type == _TERRAN_FLYING_FACTORY).nonzero()
                    target = [-1,-1]
                    if len(flyingBa_y) > 0:
                        i = random.randint(0, len(flyingBa_y) - 1)
                        target = [flyingBa_x[i], flyingBa_y[i]]
                        self.doNothingAction = DONOTHING_ACTION_LAND_BARRACKS
                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

                    elif len(flyingFa_y) > 0:
                        i = random.randint(0, len(flyingFa_y) - 1)
                        target = [flyingFa_x[i], flyingFa_y[i]]
                        self.doNothingAction = DONOTHING_ACTION_LAND_FACTORY
                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

                    elif obs.observation['player'][_IDLE_WORKER_COUNT] > 0:
                        self.doNothingAction = DONOTHING_ACTION_IDLE_WORKER
                        return actions.FunctionCall(_SELECT_IDLE_WORKER, [_SELECT_ALL])
                        
                        
                elif IsBuildAction(action):
                    # select scv
                    unit_y, unit_x = (self.unit_type == _TERRAN_SCV).nonzero()
                        
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]
                        
                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                elif action == ID_ACTION_TRAIN_ARMY:
                    if self.nextTrainArmyBuilding == _TERRAN_BARRACKS:
                        buildingType = _TERRAN_BARRACKS
                    else:
                        buildingType = _TERRAN_FACTORY

                    self.UpdateScreenBuildings(buildingType)
                    count = len(self.buildingVec[buildingType]) - 1
                    if count > 0: 
                        i = random.randint(1, count)
                        building = self.buildingVec[buildingType][i]

                        self.lastBuildingIdx = i
                        self.currentBuildingTypeSelected = buildingType
                        target = building.m_screenLocation

                        if random.randint(0, 1) == 0:
                            trainArmy = building.m_buildingAddition
                        else:
                            trainArmy = True

                        if trainArmy:
                            self.IsMultiSelect = True
                            return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, SwapPnt(target)])
                        else:
                            return actions.FunctionCall(_SELECT_POINT, [_SELECT_SINGLE, SwapPnt(target)])
  
            elif self.move_number == 1:
                self.move_number += 1
                
                action = self.previous_action
                if action == ID_ACTION_DO_NOTHING:   
                    if self.doNothingAction == DONOTHING_ACTION_LAND_BARRACKS:
                        target = self.GetLocationForBuildingAddition(obs, _TERRAN_BARRACKS)
                        if target[Y_IDX] >= 0:
                            return actions.FunctionCall(_SELECT_POINT, [_SELECT_SINGLE, SwapPnt(target)])

                    if self.doNothingAction == DONOTHING_ACTION_LAND_FACTORY:
                        target = self.GetLocationForBuildingAddition(obs, _TERRAN_FACTORY)
                        if target[Y_IDX] >= 0:
                            return actions.FunctionCall(_SELECT_POINT, [_SELECT_SINGLE, SwapPnt(target)])

                    if self.doNothingAction == DONOTHING_ACTION_IDLE_WORKER:
                        if _HARVEST_GATHER in obs.observation['available_actions']:
                            target = self.GatherHarvest()
                            if target[0] >= 0:
                                return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, SwapPnt(target)])
                    
                elif IsBuildAction(action):
                    buildingType = ACTION_2_BUILDING_TRANSITION[action]
                    sc2Action = BUIILDING_2_SC2ACTIONS[buildingType]
                    if sc2Action in obs.observation['available_actions']:

                        if action != ID_ACTION_BUILD_REFINERY:
                            coord = self.GetLocationForBuilding(obs, buildingType)
                        else:
                            coord = self.GetLocationForOilRefinery(obs)

                        if coord[Y_IDX] >= 0:
                            self.buildCommands[buildingType].append(BuildingCmd(coord))
                            return actions.FunctionCall(sc2Action, [_NOT_QUEUED, SwapPnt(coord)])

                elif action == ID_ACTION_TRAIN_ARMY:
                    self.move_number = 0
                    buildingIdx = self.lastBuildingIdx
                    self.lastBuildingIdx = -1
                              
                    if self.nextTrainArmyBuilding == _TERRAN_BARRACKS:
                        if not self.IsMultiSelect and buildingIdx > 0:
                            if _BUILD_REACTOR in obs.observation['available_actions']:
                                target = self.GetLocationForBuildingAddition(obs, _TERRAN_BARRACKS, buildingIdx)
                                if target[Y_IDX] >= 0:
                                    self.lastTrainArmyReward = 0
                                    self.nextTrainArmyBuilding = _TERRAN_FACTORY
                                    self.buildingVec[_TERRAN_BARRACKS][buildingIdx].m_buildingAddition = True
                                    return actions.FunctionCall(_BUILD_REACTOR, [_QUEUED, SwapPnt(target)])


                        if _TRAIN_REAPER in obs.observation['available_actions']:
                            self.lastTrainArmyReward = REWARD_TRAIN_REAPER / NORMALIZED_REWARD
                            self.nextTrainArmyBuilding = _TERRAN_FACTORY
                            return actions.FunctionCall(_TRAIN_REAPER, [_QUEUED])

                        if _TRAIN_MARINE in obs.observation['available_actions']:
                            self.lastTrainArmyReward = REWARD_TRAIN_MARINE / NORMALIZED_REWARD                            
                            self.nextTrainArmyBuilding = _TERRAN_FACTORY
                            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

                    else:
                        if not self.IsMultiSelect and buildingIdx > 0:
                            if _BUILD_TECHLAB in obs.observation['available_actions']:
                                target = self.GetLocationForBuildingAddition(obs, _TERRAN_FACTORY, buildingIdx)
                                if target[Y_IDX] >= 0:
                                    self.lastTrainArmyReward = 0
                                    self.buildingVec[_TERRAN_FACTORY][buildingIdx].m_buildingAddition = True
                                    self.nextTrainArmyBuilding = _TERRAN_BARRACKS
                                    return actions.FunctionCall(_BUILD_TECHLAB, [_QUEUED, SwapPnt(target)])
                        
                        if _TRAIN_SIEGE_TANK in obs.observation['available_actions']:
                            self.lastTrainArmyReward = REWARD_TRAIN_SIEGE_TANK / NORMALIZED_REWARD
                            self.nextTrainArmyBuilding = _TERRAN_BARRACKS
                            return actions.FunctionCall(_TRAIN_SIEGE_TANK, [_QUEUED])

                        if _TRAIN_HELLION in obs.observation['available_actions']:
                            self.lastTrainArmyReward = REWARD_TRAIN_HELLION / NORMALIZED_REWARD                            
                            self.nextTrainArmyBuilding = _TERRAN_BARRACKS
                            return actions.FunctionCall(_TRAIN_HELLION, [_QUEUED])
                        
                    self.lastTrainArmyReward = 0
                              
            elif self.move_number == 2:
                self.move_number = 0
                
                action = self.previous_action
                if IsBuildAction(action):
                    if _HARVEST_GATHER in obs.observation['available_actions']:
                        target = self.GatherHarvest()
                        if target[0] >= 0:
                            return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, SwapPnt(target)])

            self.actionSucceed = False
            target = self.DoNothingBuidingCheck()
            if target[0] >= 0:
                self.IsMultiSelect = True
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, SwapPnt(target)]) 

            return actions.FunctionCall(_NO_OP, [])
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def FirstStep(self, obs):
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        self.previous_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.move_number = 0
        self.current_state_for_decision_making = None
        
        for key in BUILDING_2_STATE_TRANSITION.keys():
            self.buildingVec[key] = [0]
            self.buildCommands[key] = []

        commandCenterLoc_y, commandCenterLoc_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()

        middleCC = FindMiddle(commandCenterLoc_y, commandCenterLoc_x)
        self.buildingVec[_TERRAN_COMMANDCENTER][0] += 1
        self.buildingVec[_TERRAN_COMMANDCENTER].append(BuildingCoord(middleCC))

        self.skipSupplyBuildings = True

    def LastStep(self, obs):
        self.qTable.q_table.to_pickle(Q_TABLE_FILE + '.gz', 'gzip') 

        self.move_number = 0
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.current_state_for_decision_making = None
        self.step_num = 0 
        self.doNothingAction = DONOTHING_ACTION_NOTHING

        self.accumulatedReward += self.lastTrainArmyReward
        reward = self.accumulatedReward - 1

        self.qTable.learn(str(self.previous_scaled_state), self.previous_action, reward, 'terminal')
        self.qTable.end_run(reward)

        self.lastTrainArmyReward = 0
        self.accumulatedReward = 0

    def Insert2QTable(self):
        action = self.previous_action
        if action != ID_ACTION_TRAIN_ARMY:
            self.qTable.learn(str(self.previous_scaled_state), self.previous_action, 0, str(self.current_scaled_state))
        else:
            self.accumulatedReward += self.lastTrainArmyReward
            self.qTable.learn(str(self.previous_scaled_state), self.previous_action, self.lastTrainArmyReward, str(self.current_scaled_state))

    def CreateState(self, obs):
        for key, value in BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.buildingVec[key][0]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.buildCommands[key])
     
        self.current_state[STATE_MINERALS_IDX] = obs.observation['player'][_MINERALS]
        self.current_state[STATE_GAS_IDX] = obs.observation['player'][_VESPENE]
        if CREATE_DETAILED_STATE:
            self.CreateDetailedState(obs)

        self.ScaleCurrState()

        
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[STATE_MINERALS_IDX] = math.ceil(self.current_scaled_state[STATE_MINERALS_IDX] / STATE_MINERALS_BUCKETING) * STATE_MINERALS_BUCKETING
        self.current_scaled_state[STATE_MINERALS_IDX] = min(STATE_MINERALS_MAX, self.current_scaled_state[STATE_MINERALS_IDX])
        self.current_scaled_state[STATE_GAS_IDX] = math.ceil(self.current_scaled_state[STATE_GAS_IDX] / STATE_GAS_BUCKETING) * STATE_GAS_BUCKETING
        self.current_scaled_state[STATE_GAS_IDX] = min(STATE_GAS_MAX, self.current_scaled_state[STATE_GAS_IDX])

    def CreateDetailedState(self, obs):       
        return

    def GatherHarvest(self):
        if random.randint(0, 4) < 4:
            resourceList = _NEUTRAL_MINERAL_FIELD
        else:
            resourceList = [_TERRAN_OIL_REFINERY]

        unit_y = []
        unit_x = []
        for val in resourceList[:]:
            p_y, p_x = (self.unit_type == val).nonzero()
            unit_y += list(p_y)
            unit_x += list(p_x)

        if len(unit_y) > 0:
            i = random.randint(0, len(unit_y) - 1)
            return [unit_y[i], unit_x[i]]
        
        return [-1,-1]


    def UpdateSupplyCap(self, obs):
        supplyCap = obs.observation['player'][_SUPPLY_CAP]
        stateSupplyCap = self.ComputeSupplyCap()
        supplyBuildingFiguredOut = True
        # if supplyCap != stateSupplyCap:
        #     diff = supplyCap - stateSupplyCap
        #     if diff > 0:
        #         numSDCmd = 0
        #         numCCCmd = 0
        #         for itr in self.buildCommands[:]:
        #             if itr.m_buildingType == _TERRAN_SUPPLY_DEPOT:
        #                 numSDCmd += 1
        #             elif itr.m_buildingType == _TERRAN_COMMANDCENTER:
        #                 numCCCmd += 1

        #         if numSDCmd > 0:
        #             if numCCCmd == 0:
        #                 if diff % SUPPLY_SIZE_SD == 0:
        #                     for i in range(0 , int(diff / SUPPLY_SIZE_SD)):
        #                         self.LearnEndBuilding(_TERRAN_SUPPLY_DEPOT)
        #                 else:
        #                     supplyBuildingFiguredOut = False

        #             else:
        #                 supplyBuildingFiguredOut = False
        #         elif numCCCmd > 0:
        #             if diff % SUPPLY_SIZE_CC == 0:
        #                 for i in range(0 , int(diff / SUPPLY_SIZE_CC)):
        #                     self.LearnEndBuilding(_TERRAN_COMMANDCENTER)
        #             else:
        #                 supplyBuildingFiguredOut = False

        #     else:
        #          supplyBuildingFiguredOut = False

        # self.skipSupplyBuildings = supplyBuildingFiguredOut
    
    def ComputeSupplyCap(self):
        numCC = self.buildingVec[_TERRAN_COMMANDCENTER][0]
        numSD = self.buildingVec[_TERRAN_SUPPLY_DEPOT][0]
        return SUPPLY_SIZE_SD * numSD + SUPPLY_SIZE_CC * numCC

    def UpdateBuildingInProgress(self, obs):
        #sort command by in progress
        def getKey(cmd):
            return cmd.m_inProgress

        for building, commands in self.buildCommands.items():
            for cmd in commands[:]:
                if not cmd.m_inProgress:
                    x = cmd.m_screenLocation[X_IDX]
                    y = cmd.m_screenLocation[Y_IDX]

                    for off_y in range (-1, 1):
                        for off_x in range (-1, 1):
                            if self.unit_type[y + off_y][x + off_x] == building:
                                cmd.m_inProgress = True
                                break
                        if cmd.m_inProgress:
                            break

                    cmd.m_steps2Check -= 1
                    if cmd.m_steps2Check == 0:
                        # print("\tbuild removed from commands", BUILDING_NAMES[building])
                        self.buildCommands[building].remove(cmd)

            self.buildCommands[building].sort(key = getKey, reverse=True)


    def UpdateBuildingCompletion(self, obs, buildingType):
        buildingStatus = obs.observation['multi_select']
        if len(buildingStatus) == 0:
            buildingStatus = obs.observation['single_select']

        numComplete = 0
        inProgress = 0
        for stat in buildingStatus[:]:
            if stat[_BUILDING_COMPLETION_IDX] == 0:
                numComplete += 1
            else:
                inProgress += 1

        vecInProgress = 0
        for buildingCmd in self.buildCommands[buildingType][:]:
            if buildingCmd.m_inProgress:
                vecInProgress += 1
        
        numBuildingsInVec = self.buildingVec[buildingType][0]

        diff = numComplete - numBuildingsInVec
        # add new buildings
        if diff > 0:
            # specific error reason unknown
            if numBuildingsInVec == 0 and numComplete == 12:
                return
            
            buildingFinished = 0
            for building in self.buildCommands[buildingType][:]:
                if building.m_inProgress:
                    buildingFinished += 1
                    self.buildCommands[buildingType].remove(building)
                
                if buildingFinished == diff:
                    break
            
            numBuildingsInVec += buildingFinished
            diff = numComplete - numBuildingsInVec
            # if diff > 0 :
            #     print("\n\nError in check if building finished!!")

            self.buildingVec[buildingType][0] = numComplete
        elif diff < 0:

            # remove destroyed buildings
            self.buildingVec[buildingType][0] = numComplete

    def DoNothingBuidingCheck(self):
        self.donothingBuildingIdx = (self.donothingBuildingIdx + 1) % len(DO_NOTHING_BUILDING_CHECK)
        self.currentBuildingTypeSelected = DO_NOTHING_BUILDING_CHECK[self.donothingBuildingIdx]
        self.UpdateScreenBuildings(self.currentBuildingTypeSelected)
        numScreenBuilding = len(self.buildingVec[self.currentBuildingTypeSelected]) - 1
        if numScreenBuilding > 0:
            idxBuilding = random.randint(1, numScreenBuilding)
            coord = self.buildingVec[self.currentBuildingTypeSelected][idxBuilding].m_screenLocation
            return coord
        else:
            return [-1,-1]
        
    def UpdateScreenBuildings(self, buildingType):
        self.buildingVec[buildingType] = [self.buildingVec[buildingType][0]]
        buildingMat = self.unit_type == buildingType
        pnt_y, pnt_x = buildingMat.nonzero()
        if len (pnt_y) > 0:
            idxBuilding = 0
            buildings = []
            while len(pnt_y) > 0:
                building_y, building_x = IsolateArea([pnt_y[0], pnt_x[0]], buildingMat)
                toRemove = []
                for matPnt in range(0, len(pnt_y)):
                    found = False
                    for buildPnt in range(0, len(building_y)):
                        if pnt_y[matPnt] == building_y[buildPnt] and pnt_x[matPnt] == building_x[buildPnt]:
                            found = True
                            break
                    
                    if found:
                        toRemove.append(matPnt)

                pnt_y = np.delete(pnt_y, toRemove)
                pnt_x = np.delete(pnt_x, toRemove)

                coord = FindMiddle(building_y, building_x)
                hasAddition = self.HasBuildingAddition(buildingType, building_y, building_x)
                # wrong assumption : building is built
                buildings.append(BuildingCoord(coord, hasAddition))
                idxBuilding += 1

            for idx in range (0, len(buildings)):
                self.buildingVec[buildingType].append(buildings[idx])

    def HasBuildingAddition(self, buildingType, building_y, building_x):
        if buildingType == _TERRAN_BARRACKS:
            addition = _TERRAN_REACTOR
        elif buildingType == _TERRAN_FACTORY:
            addition = _TERRAN_TECHLAB
        else:
            return False

        additionMat = self.unit_type == addition
        for i in range(0, len(building_y)):
            nearX = building_x[i] + 1
            if nearX < SCREEN_SIZE[X_IDX] and additionMat[building_y[i]][nearX]:
                return True

        return False

    def GetLocationForBuilding(self, obs, buildingType):
        neededSize = BUILDING_SIZES[buildingType]
        cameraHeightMap = self.CreateCameraHeightMap(obs)
        occuppyMat = self.unit_type > 0

        foundLoc = False
        location = [-neededSize, -neededSize]
        for y in range(0, SCREEN_SIZE[Y_IDX] - neededSize):
            for x in range(0, SCREEN_SIZE[X_IDX] - neededSize):
                foundLoc = self.HaveSpace(occuppyMat, cameraHeightMap, y, x, neededSize)
                    
                if foundLoc:
                    location = [y, x]
                    break

            if foundLoc:
                break

        location[Y_IDX] += int(neededSize / 2)
        location[X_IDX] += int(neededSize / 2)

        return location

    def GetLocationForBuildingAddition(self, obs, buildingType, buildingIdx = -1):
        neededSize = BUILDING_SIZES[buildingType]
        additionSize = BUILDING_SIZES[_TERRAN_REACTOR]
        
        cameraHeightMap = self.CreateCameraHeightMap(obs)
        occuppyMat = self.unit_type > 0
        
        if buildingIdx >= 0:
            dfltPnt = self.buildingVec[buildingType][buildingIdx].m_screenLocation
            # find right edge of building
            y, x = self.FindBuildingRightEdge(buildingType,dfltPnt)
            if y < SCREEN_SIZE[Y_IDX] and x < SCREEN_SIZE[X_IDX] and self.HaveSpace(occuppyMat, cameraHeightMap, y, x, additionSize):
                return dfltPnt

        foundLoc = False
        location = [-1, -1]
        for y in range(0, SCREEN_SIZE[Y_IDX] - neededSize):
            for x in range(0, SCREEN_SIZE[X_IDX] - neededSize - additionSize):
                if self.HaveSpace(occuppyMat, cameraHeightMap, y, x, neededSize):
                    additionY = y + int((neededSize / 2) - (additionSize / 2))
                    additionX = x + neededSize
                    foundLoc = self.HaveSpace(occuppyMat, cameraHeightMap, additionY, additionX, additionSize)
                    
                if foundLoc:
                    location = [y + int(neededSize / 2), x + int(neededSize / 2)]
                    break

            if foundLoc:
                break

        return location

    def GetLocationForOilRefinery(self, obs):
        refMat = self.unit_type == _TERRAN_OIL_REFINERY
        ref_y,ref_x = refMat.nonzero()
        gasMat = self.unit_type == _VESPENE_GAS_FIELD
        vg_y, vg_x = gasMat.nonzero()


        if len(vg_y) == 0:
            return [-1, -1]
        
        if len(ref_y) == 0:
            # no refineries
            location = vg_y[0], vg_x[0]
            vg_y, vg_x = IsolateArea(location, gasMat)
            midPnt = FindMiddle(vg_y, vg_x)
            return midPnt
        else:
            rad2Include = 4

            initLoc = False
            for pnt in range(0, len(vg_y)):
                found = False
                i = 0
                while not found and i < len(ref_y):
                    if abs(ref_y[i] - vg_y[pnt]) < rad2Include and abs(ref_x[i] - vg_x[pnt]) < rad2Include:
                        found = True
                    i += 1

                if not found:
                    initLoc = True
                    location = vg_y[pnt], vg_x[pnt]
                    break
            
            if initLoc:
                newVG_y, newVG_x = IsolateArea(location, gasMat)
                midPnt = FindMiddle(newVG_y, newVG_x)
                return midPnt

        return [-1, -1]

    def HaveSpace(self, occuppyMat, heightsMap, yStart, xStart, neededSize):
        height = heightsMap[yStart][xStart]
        if height == 0:
            return False

        yEnd = min(yStart + neededSize, SCREEN_SIZE[Y_IDX])
        xEnd = min(xStart + neededSize, SCREEN_SIZE[X_IDX])
        for y in range (yStart, yEnd):
            for x in range (xStart, xEnd):
                if occuppyMat[y][x] or height != heightsMap[y][x]:
                    return False
        
        return True
    
    def FindBuildingRightEdge(self, buildingType, point):
        buildingMat = self.unit_type == buildingType
        found = False
        x = point[X_IDX]
        y = point[Y_IDX]

        while not found:
            if x + 1 >= SCREEN_SIZE[X_IDX]:
                break 

            x += 1
            if not buildingMat[y][x]:
                if y + 1 < SCREEN_SIZE[Y_IDX] and buildingMat[y + 1][x]:
                    y += 1
                elif y > 0 and buildingMat[y - 1][x]:
                    y -= 1
                else:
                    found = True

        return y,x
    def CreateCameraHeightMap(self,obs):
        height_map = obs.observation['minimap'][_HEIGHT_MAP]
        cameraHeightMap = np.zeros((SCREEN_SIZE[Y_IDX], SCREEN_SIZE[X_IDX]), dtype=int)
        for y in range(0, SCREEN_SIZE[Y_IDX]):
            for x in range(0, SCREEN_SIZE[X_IDX]):
                point = [y, x]
                scaledPnt = Scale2MiniMap(point, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                cameraHeightMap[y][x] = height_map[scaledPnt[Y_IDX]][scaledPnt[X_IDX]]

        return cameraHeightMap
              
    def PrintMiniMap(self, obs):
        selfPnt_y, selfPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        for y in range(MINI_MAP_START[Y_IDX], MINI_MAP_END[Y_IDX]):
            for x in range(MINI_MAP_START[X_IDX], MINI_MAP_END[X_IDX]):
                isSelf = False
                for i in range (0, len(selfPnt_y)):
                    if (y == selfPnt_y[i] and x == selfPnt_x[i]):
                        isSelf = True
                
                isEnemy = False
                for i in range (0, len(enemyPnt_y)):
                    if (y == enemyPnt_y[i] and x == enemyPnt_x[i]):
                        isEnemy = True

                if (x == self.cameraCornerNorthWest[X_IDX] and y == self.cameraCornerNorthWest[Y_IDX]) or (x == self.cameraCornerSouthEast[X_IDX] and y == self.cameraCornerSouthEast[Y_IDX]):
                    print ('#', end = '')
                elif isSelf:
                    print ('s', end = '')
                elif isEnemy:
                    print ('e', end = '')
                else:
                    print ('_', end = '')
            print('|')  

    def PrintScreen(self, valToPrint = -1, addPoints = []):
        nonPrintedVals = []
        for y in range(0, SCREEN_SIZE[Y_IDX]):
            for x in range(0, SCREEN_SIZE[X_IDX]):        
                foundInPnts = False
                for i in range(0, len (addPoints)):
                    if addPoints[i][X_IDX] == x and addPoints[i][Y_IDX] == y:
                        foundInPnts = True

                uType = self.unit_type[y][x]
                if foundInPnts:
                    print (' ', end = '')
                elif uType == valToPrint:
                    print ('V', end = '')
                elif uType in UNIT_CHAR:
                    print(UNIT_CHAR[uType], end = '')
                else:
                    if uType > 0 and uType not in nonPrintedVals:
                        nonPrintedVals.append(uType)
                    print ('_', end = '')
            print('|') 
    
        if len(nonPrintedVals) > 0:
            print("non printed vals = ", nonPrintedVals) 
            time.sleep(2)
            self.SearchNewBuildingPnt()

    def PrintScreenTest(self):
        vg_y, vg_x = (self.unit_type == _VESPENE_GAS_FIELD).nonzero()
        
        for y in range(0, SCREEN_SIZE[Y_IDX]):
            for x in range(0, SCREEN_SIZE[X_IDX]):
                for i in range(0, len (vg_y)):
                    printed = False
                    if vg_x[i] == x and vg_y[i] == y:
                        print ('g', end = '')
                        printed = True

                if printed:
                    continue

                print ('_', end = '')
            print('|') 
    def SearchNewBuildingPnt(self):
        print("search new building point")
        for i in range(1, 100):
            if i not in UNIT_CHAR:
                pnts_y,pnts_x = (self.unit_type == i).nonzero()
                if len(pnts_y) > 0:
                    self.PrintScreen(i)
                    print("exist idx =", i, "\n\n\n")
