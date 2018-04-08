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

# multi and single select information
_BUILDING_TRNSPORT_SLOT = 5
_BUILDING_COMPLETION_IDX = 6

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_OIL_REFINERY = 20
_TERRAN_BARRACKS = 21
_TERRAN_FACTORY = 27
_TERRAN_REACTOR = 38
_TERRAN_TECHLAB = 39

_TERRAN_FLYING_BARRACKS = 33
_TERRAN_FLYING_FACTORY = 100

_NEUTRAL_MINERAL_FIELD = 341
_VESPENE_GAS_FIELD = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_SINGLE = [0]
_SELECT_ALL = [2]

COMMAND_CENTER_SCREEN_SIZE = 18
SUPPLY_DEPOT_SCREEN_SIZE = 9
BARRACKS_SCREEN_SIZE = 12
FACTORY_SCREEN_SIZE = 12

REACTOR_SCREEN_SIZE = 3
REACTOR_SCREEN_LOCATION = [-1 ,-1]

Q_TABLE_FILE = 'buildbase_q_table'

ID_ACTION_DO_NOTHING = 0
ID_ACTION_BUILD_SUPPLY_DEPOT = 1
ID_ACTION_BUILD_BARRACKS = 2
ID_ACTION_BUILD_REFINERY = 3
ID_ACTION_BUILD_FACTORY = 4
ID_ACTION_TRAIN_ARMY = 5

NUM_ACTIONS = 6

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
STATE_SUPPLY_DEPOT_IDX = 1
STATE_REFINERY_IDX = 2
STATE_BARRACKS_IDX = 3
STATE_FACTORY_IDX = 4
STATE_MINERALS_IDX = 5
STATE_GAS_IDX = 6
STATE_SIZE = 7

REWARD_TRAIN_MARINE = 1
REWARD_TRAIN_REAPER = 2
REWARD_TRAIN_HELLION = 4
REWARD_TRAIN_SIEGE_TANK = 6
NORMALIZED_REWARD = 400

CREATE_DETAILED_STATE = False
DETAILED_STATE_SIZE = 6

class BuildingCoord:
    def __init__(self, buildingType, screenLocation):
        self.m_buildingType = buildingType
        self.m_screenLocation = screenLocation


class AttackCmd:
    def __init__(self, state, attackCoord, isBaseAttack ,attackStarted = False):
        self.m_state = state
        self.m_attackCoord = [math.ceil(attackCoord[Y_IDX]), math.ceil(attackCoord[X_IDX])]
        self.m_inTheWayBattle = [-1, -1]
        self.m_isBaseAttack = isBaseAttack
        self.m_attackStarted = attackStarted
        self.m_attackEnded = False

class BuildingCmd(BuildingCoord):
    def __init__(self, state, buildingType, screenLocation):
        BuildingCoord.__init__(self, buildingType, screenLocation)
        self.m_state = state

class BuildingDetails(BuildingCoord):
    def __init__(self, buildingType, screenLocation, minimapLocation):
        BuildingCoord.__init__(self, buildingType, screenLocation)
        self.m_minimapLocation = minimapLocation
        self.m_buildingFinished = False
        self.m_buildingAddition = False




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

def Flood(location, buildingMap):   
    closeLocs = [[location[Y_IDX] + 1, location[X_IDX]], [location[Y_IDX] - 1, location[X_IDX]], [location[Y_IDX], location[X_IDX] + 1], [location[Y_IDX], location[X_IDX] - 1] ]
    points_y = [location[Y_IDX]]
    points_x = [location[X_IDX]]
    for loc in closeLocs[:]:
        if buildingMap[loc[Y_IDX]][loc[X_IDX]]:
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

    print ("from point = ", point, "scaled point = ", scaledPoint)
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
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 5
        self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 5

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
        
        numTotalRuns = self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot]
        numExpRuns = self.q_table.ix[self.TrialsData, self.NumRunsExperimentSlot]
        
        totReward = self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot]
        expReward = self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot]

        totReward = (numTotalRuns * totReward + r) / (numTotalRuns + 1)
        expReward = (numExpRuns * expReward + r) / (numExpRuns + 1)

        self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot] = totReward
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = expReward

        
        self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot] += 1
        self.q_table.ix[self.TrialsData, self.AvgRewardExperimentSlot] += 1

        print("num total runs = ", numTotalRuns, "avg total = ", totReward)
        print("num experiment runs = ", numExpRuns, "avg experiment = ", expReward)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class BuildBase(base_agent.BaseAgent):
    def __init__(self):
        super(BuildBase, self).__init__()
       
        self.qTable = QLearningTable(actions=list(range(NUM_ACTIONS)))

        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        if CREATE_DETAILED_STATE:
            self.previous_Detailedstate = np.zeros(DETAILED_STATE_SIZE, dtype=np.int32, order='C')
            self.current_Detailedstate = np.zeros(DETAILED_STATE_SIZE, dtype=np.int32, order='C')

        self.previous_action = None
        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]
        self.unit_type = None
        self.cc_y = None
        self.cc_x = None
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.move_number = 0
        self.current_state_for_decision_making = None
        self.step_num = 0

        self.buildingVec = []
        self.buildCommands = []

        self.topLeftBaseLocation = [23,18]
        self.bottomRightBaseLocation = [45,39]
        if os.path.isfile(Q_TABLE_FILE + '.gz'):
            self.qTable.q_table = pd.read_pickle(Q_TABLE_FILE + '.gz', compression='gzip')
        
        self.BuildingSizes = {}
        self.BuildingSizes[_TERRAN_COMMANDCENTER] = COMMAND_CENTER_SCREEN_SIZE
        self.BuildingSizes[_TERRAN_SUPPLY_DEPOT] = SUPPLY_DEPOT_SCREEN_SIZE
        self.BuildingSizes[_TERRAN_BARRACKS] = BARRACKS_SCREEN_SIZE
        self.BuildingSizes[_TERRAN_FACTORY] = FACTORY_SCREEN_SIZE
        self.BuildingSizes[_TERRAN_REACTOR] = REACTOR_SCREEN_SIZE
        
        self.lastTrainArmyReward = 0
        self.currentBuildingType = 0
        self.nextTrainArmyBuilding = _TERRAN_BARRACKS
        self.accumulatedReward = 0
        self.lastBuildingCoord = [-1,-1]
        self.lastBuildingIdx = -1
        self.IsMultiSelect = False

        self.bool = False

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

            self.AddNewBuilding(obs)

            if self.move_number == 0:
                # PrintBuildingSizes(self.unit_type)
                if self.bool == True:
                    self.PrintScreen()
                    time.sleep(3)
                self.CreateState(obs)
                
                self.move_number += 1
                action = self.qTable.choose_action(str(self.current_scaled_state))
                if action == ID_ACTION_BUILD_SUPPLY_DEPOT and self.current_state[STATE_SUPPLY_DEPOT_IDX] > 10:
                    action = ID_ACTION_DO_NOTHING

                if self.previous_action is not None:
                    if action != ID_ACTION_TRAIN_ARMY:
                        self.qTable.learn(str(self.previous_scaled_state), self.previous_action, 0, str(self.current_scaled_state))
                    else:
                        self.accumulatedReward += self.lastTrainArmyReward
                        self.qTable.learn(str(self.previous_scaled_state), self.previous_action, self.lastTrainArmyReward, str(self.current_scaled_state))

                self.previous_state[:] = self.current_state[:]
                self.previous_scaled_state[:] = self.current_scaled_state[:]
                self.previous_action = action

                if action == ID_ACTION_DO_NOTHING:
                    # monitor destroyed buildings
                    idxBuilding = -1
                    i = random.randint(0, 2)
                    if i == 0 and self.current_state[STATE_COMMAND_CENTER_IDX] > 0:
                        idxBuilding = random.randint(0, self.current_state[STATE_COMMAND_CENTER_IDX] - 1)
                        self.currentBuildingType = _TERRAN_COMMANDCENTER
                    elif i == 1 and self.current_state[STATE_SUPPLY_DEPOT_IDX] > 0:
                        idxBuilding = random.randint(0, self.current_state[STATE_SUPPLY_DEPOT_IDX] - 1)
                        self.currentBuildingType = _TERRAN_SUPPLY_DEPOT
                    elif  self.current_state[STATE_REFINERY_IDX] > 0:
                        idxBuilding = random.randint(0, self.current_state[STATE_REFINERY_IDX] - 1)
                        self.currentBuildingType = _TERRAN_OIL_REFINERY

                    target = [-1,-1]
                    idx = 0
                    for building in self.buildingVec[:]:
                        if building.m_buildingType == self.currentBuildingType:
                            idxBuilding -= 1
                        if idxBuilding < 0:
                            target = building.m_screenLocation
                            break
                        idx += 1
                    if target[Y_IDX] >= 0:
                        self.IsMultiSelect = True
                        return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, SwapPnt(target)]) 

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
                        count = self.current_state[STATE_BARRACKS_IDX]
                    else:
                        buildingType = _TERRAN_FACTORY
                        count = self.current_state[STATE_FACTORY_IDX]

                    if count > 0:                         
                        i = random.randint(0, count - 1)

                        target = [-1,-1]
                        idx = 0
                        for building in self.buildingVec[:]:
                            if building.m_buildingType == buildingType:
                                i -= 1
                            if i < 0:
                                target = building.m_screenLocation
                                if random.randint(0, 1) == 0:
                                    trainArmy = building.m_buildingAddition
                                else:
                                    trainArmy = True
                                break
                            idx += 1

                        self.lastBuildingCoord = target
                        self.lastBuildingIdx = idx
                        if trainArmy:
                            self.IsMultiSelect = True
                            return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, SwapPnt(target)])
                        else:
                            return actions.FunctionCall(_SELECT_POINT, [_SELECT_SINGLE, SwapPnt(target)])
 
            
            elif self.move_number == 1:
                self.move_number += 1
                
                action = self.previous_action
                if action == ID_ACTION_DO_NOTHING:
                    if self.IsMultiSelect:
                        self.UpdateBuildingCompletion(obs, self.currentBuildingType)
                        self.IsMultiSelect = False
                elif action == ID_ACTION_BUILD_SUPPLY_DEPOT: 
                    if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                        coord = self.GetLocationForBuilding(obs, self.BuildingSizes[_TERRAN_SUPPLY_DEPOT])
                        if coord[Y_IDX] >= 0:
                            self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_SUPPLY_DEPOT, coord))
                            return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, SwapPnt(coord)])
                
                elif action == ID_ACTION_BUILD_BARRACKS:
                    if _BUILD_BARRACKS in obs.observation['available_actions']:
                        coord = self.GetLocationForBuilding(obs, self.BuildingSizes[_TERRAN_BARRACKS])
                        if coord[Y_IDX] >= 0:
                            self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_BARRACKS, coord))
                            return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, SwapPnt(coord)])

                elif action == ID_ACTION_BUILD_REFINERY:
                    refCount = self.current_state[STATE_REFINERY_IDX]
                    if _BUILD_OIL_REFINERY in obs.observation['available_actions']:
                        if self.cc_y.any():
                            target = self.GetLocationForOilRefinery(obs, refCount)
                            if target[0] != -1:
                                self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_OIL_REFINERY, target))
                                return actions.FunctionCall(_BUILD_OIL_REFINERY, [_NOT_QUEUED, SwapPnt(target)])
                elif action == ID_ACTION_BUILD_FACTORY:
                    if _BUILD_FACTORY in obs.observation['available_actions']:
                        coord = self.GetLocationForBuilding(obs, self.BuildingSizes[_TERRAN_FACTORY])
                        if coord[Y_IDX] >= 0:
                            self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_FACTORY, coord))
                            return actions.FunctionCall(_BUILD_FACTORY, [_NOT_QUEUED, SwapPnt(coord)])

                elif action == ID_ACTION_TRAIN_ARMY:
                    if self.IsMultiSelect:
                        self.UpdateBuildingCompletion(obs, self.nextTrainArmyBuilding)
                        self.IsMultiSelect = False

                    buildingIdx = self.lastBuildingIdx
                    target = self.lastBuildingCoord
                    self.lastBuildingCoord = [-1,-1]
                    self.lastBuildingIdx = -1
                              
                    if self.nextTrainArmyBuilding == _TERRAN_BARRACKS:
                        if _BUILD_REACTOR in obs.observation['available_actions']:
                            target = self.GetLocationForBuildingAddition(obs, self.BuildingSizes[self.nextTrainArmyBuilding], target)
                            if target[Y_IDX] > 0:
                                self.bool = True
                                self.lastTrainArmyReward = 0
                                self.nextTrainArmyBuilding = _TERRAN_FACTORY
                                self.buildingVec[buildingIdx].m_buildingAddition = True
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

                        if _BUILD_TECHLAB in obs.observation['available_actions'] and target[0] >= 0:
                            self.lastTrainArmyReward = 0
                            self.buildingVec[buildingIdx].m_buildingAddition = True
                            self.nextTrainArmyBuilding = _TERRAN_BARRACKS
                            return actions.FunctionCall(_BUILD_TECHLAB, [_QUEUED, SwapPnt(target)])
                        
                        # print("try factory army training")
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
                        if random.randint(0, 4) < 4:
                            unit_y, unit_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                        else:
                            unit_y, unit_x = (self.unit_type == _TERRAN_OIL_REFINERY).nonzero()

                        if unit_y.any():
                            i = random.randint(0, len(unit_y) - 1)
                            
                            m_x = unit_x[i]
                            m_y = unit_y[i]
                            
                            target = [int(m_x), int(m_y)]
                            
                            return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
                
                return actions.FunctionCall(_NO_OP, [])
            return actions.FunctionCall(_NO_OP, [])
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def FirstStep(self, obs):
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        self.cc_y, self.cc_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()

        self.previous_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.move_number = 0
        self.current_state_for_decision_making = None
        
        commandCenterLoc_y, commandCenterLoc_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()

        miniMapLoc = [-1, -1]
        if (commandCenterLoc_y.any()):
            middleCC = FindMiddle(commandCenterLoc_y, commandCenterLoc_x)
            miniMapLoc = Scale2MiniMap(middleCC, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
        else:
            exit(1)

        cc = BuildingDetails(_TERRAN_COMMANDCENTER, middleCC, miniMapLoc)
        self.buildingVec.append(cc)

    def LastStep(self, obs):
        self.qTable.q_table.to_pickle(Q_TABLE_FILE + '.gz', 'gzip') 

        self.move_number = 0
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.current_state_for_decision_making = None
        self.cc_y = None
        self.cc_x = None
        self.step_num = 0 

        self.accumulatedReward += self.lastTrainArmyReward
        reward = self.accumulatedReward - 1

        self.qTable.learn(str(self.previous_scaled_state), self.previous_action, reward, 'terminal')
        self.qTable.end_run(reward)

        self.lastTrainArmyReward = 0
        self.accumulatedReward = 0
        self.lastBuildingCoord = [-1,-1]
        self.buildingVec.clear()
        self.buildCommands.clear()

    def CreateState(self, obs):
        self.UpdateBuildingCounts()
     
        self.current_state[STATE_MINERALS_IDX] = obs.observation['player'][_MINERALS]
        self.current_state[STATE_GAS_IDX] = obs.observation['player'][_VESPENE]
        if CREATE_DETAILED_STATE:
            self.CreateDetailedState(obs)

        self.ScaleCurrState()
        # print(self.current_scaled_state)

    def CreateDetailedState(self, obs):       
        return

    def AddNewBuilding(self, obs):
        for itr in self.buildCommands[:]:
            buildingCoord_y, buildingCoord_x = (self.unit_type == itr.m_buildingType).nonzero()
            isBuilt = False
            for i in range(0, len(buildingCoord_y)):
                if (itr.m_screenLocation[Y_IDX] == buildingCoord_y[i] and itr.m_screenLocation[X_IDX] == buildingCoord_x[i]):
                    isBuilt = True

            if isBuilt:
                buildingMap = self.unit_type == itr.m_buildingType
                loc = [int(itr.m_screenLocation[0]),int(itr.m_screenLocation[1])]
                buildingPnt_y, buildingPnt_x = IsolateArea(loc, buildingMap)
                midPnt = FindMiddle(buildingPnt_y, buildingPnt_x)

                miniMapCoord = Scale2MiniMap(midPnt, self.cameraCornerNorthWest , self.cameraCornerSouthEast)
                isExist = False
                for itr2 in self.buildingVec[:]:
                    if itr2.m_buildingType == itr.m_buildingType and itr2.m_minimapLocation == miniMapCoord:
                        isExist = True

                # add building to vector if its exist
                if not isExist:    
                    self.buildingVec.append(BuildingDetails(itr.m_buildingType, midPnt, miniMapCoord))

                
                # remove build command from vector
                self.buildCommands.remove(itr)
                
        # self.RemoveDestroyedBuildings(obs)
    def SearchNewBuildingPnt(self):
        allBuildingIdx = [_TERRAN_COMMANDCENTER,_TERRAN_SUPPLY_DEPOT, _TERRAN_OIL_REFINERY, _TERRAN_BARRACKS, _TERRAN_FACTORY, _TERRAN_REACTOR, _TERRAN_TECHLAB]

        print("search new building point")
        for i in range(1, 100):
            if i not in allBuildingIdx:
                pnts_y,pnts_x = (self.unit_type == i).nonzero()
                if len(pnts_y) > 0:
                    print("exist idx =", i)

    def SearchFlyingBuilding(self, buildingType):      
        flyingType = -1
        if buildingType == _TERRAN_BARRACKS:
            flyingType = _TERRAN_FLYING_BARRACKS
            print("flying type search barracks = ", _TERRAN_FLYING_BARRACKS)

        elif buildingType == _TERRAN_FACTORY:
            flyingType = _TERRAN_FLYING_FACTORY
            print("flying type search factory = ", _TERRAN_FLYING_BARRACKS)


        pnts_y, pnts_x = (self.unit_type == flyingType).nonzero()
        if pnts_y.any():
            return [pnts_y[0], pnts_x[0]]
        else: 
            print ("cannot find flying object")
        return []
    def RemoveDestroyedBuildings(self, obs):

        selfPnt_y, selfPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        toRemove = []
        for curr in range(0, len(self.buildingVec)):
            isDead = True
            currBuilding = self.buildingVec[curr]
            for i in range(0, len(selfPnt_y)):
                if currBuilding.m_minimapLocation[Y_IDX] == selfPnt_y[i] and currBuilding.m_minimapLocation[X_IDX] == selfPnt_x[i]:
                    isDead = False

            if not isDead:
                isDup = False
                for prev in range(0, curr):
                    if self.buildingVec[prev].m_buildingType == currBuilding.m_buildingType and self.buildingVec[prev].m_minimapLocation == currBuilding.m_minimapLocation:
                        isDup = True
                        break


            if isDead or isDup:
                if isDead and currBuilding.m_buildingType == _TERRAN_OIL_REFINERY:
                    self.PrintScreen()
                    self.PrintMiniMap(obs)
                    print("dead building oil refinery")
                    time.sleep(10)
                toRemove.append(curr)

        for i in range (0, len(toRemove)):
             self.buildingVec.remove(self.buildingVec[toRemove[i] - i])


    def UpdateBuildingCounts(self):
        ccCount = 0
        sdCount = 0
        baCount = 0
        refCount = 0
        faCount = 0
        for itr in self.buildingVec[:]:
            if itr.m_buildingType == _TERRAN_COMMANDCENTER:
                ccCount += 1
            elif itr.m_buildingType == _TERRAN_SUPPLY_DEPOT:
                sdCount += 1
            elif itr.m_buildingType == _TERRAN_BARRACKS:
                baCount += 1
            elif itr.m_buildingType == _TERRAN_OIL_REFINERY:
                refCount += 1
            elif itr.m_buildingType == _TERRAN_FACTORY:
                faCount += 1

        self.current_state[STATE_COMMAND_CENTER_IDX] = ccCount
        self.current_state[STATE_SUPPLY_DEPOT_IDX] = sdCount
        self.current_state[STATE_BARRACKS_IDX] = baCount
        self.current_state[STATE_FACTORY_IDX] = faCount
        self.current_state[STATE_REFINERY_IDX] = refCount 

    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[STATE_MINERALS_IDX] = math.ceil(self.current_scaled_state[STATE_MINERALS_IDX] / STATE_MINERALS_BUCKETING) * STATE_MINERALS_BUCKETING
        self.current_scaled_state[STATE_MINERALS_IDX] = min(STATE_MINERALS_MAX, self.current_scaled_state[STATE_MINERALS_IDX])
        self.current_scaled_state[STATE_GAS_IDX] = math.ceil(self.current_scaled_state[STATE_GAS_IDX] / STATE_GAS_BUCKETING) * STATE_GAS_BUCKETING
        self.current_scaled_state[STATE_GAS_IDX] = min(STATE_GAS_MAX, self.current_scaled_state[STATE_GAS_IDX])

    def GetLocationForBuilding(self, obs, neededSize):
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

    def GetLocationForBuildingAddition(self, obs, neededSize, defltPnt):
        print("search location for addition!!")
        additionSize = self.BuildingSizes[_TERRAN_REACTOR]

        cameraHeightMap = self.CreateCameraHeightMap(obs)
        occuppyMat = self.unit_type > 0

        found = False
        x = defltPnt[X_IDX]
        y = defltPnt[Y_IDX]
        toAdvanceY = 1
        while not found:
            if x + 1 > SCREEN_SIZE[X_IDX]:
                found = True    
            elif occuppyMat[y][x + 1]:
                x += 1
            elif occuppyMat[y + toAdvanceY][x + 1]:
                x += 1
                y += toAdvanceY
            elif occuppyMat[y - toAdvanceY][x + 1]:
                x += 1
                y -= toAdvanceY
            else:
                found = True

            if y + 1 > SCREEN_SIZE[X_IDX] or y - 1 < 0:
                toAdvanceY = 0

        x += 1

        inPlace = True
        pnts = []
        for yAdd in range(y, y + additionSize):
            for xAdd in range(x, x + additionSize):
                if occuppyMat[yAdd][xAdd]:
                    inPlace = False
                pnts.append([yAdd, xAdd])

        PrintSpecificMatAndPnt(occuppyMat, pnts)
        print("\n\nfound in place = ", inPlace)
        time.sleep(5)

        location = defltPnt
        # foundLoc = False
        # location = [-neededSize, -neededSize]
        # for y in range(0, SCREEN_SIZE[Y_IDX] - neededSize):
        #     for x in range(0, SCREEN_SIZE[X_IDX] - neededSize - additionSize):
        #         if self.HaveSpace(occuppyMat, cameraHeightMap, y, x, neededSize):
        #             additionY = y + int((neededSize / 2) - (additionSize / 2))
        #             additionX = x + neededSize
        #             foundLoc = self.HaveSpace(occuppyMat, cameraHeightMap, additionY, additionX, additionSize)
                    
        #         if foundLoc:
        #             location = [y, x]
        #             points=[]
        #             for buildY in range(0, neededSize):
        #                 for buildX in range(0, neededSize):
        #                     points.append([buildY + y, buildX + x])
        #             for addY in range(0, additionSize):
        #                 for addX in range(0, additionSize):
        #                     points.append([addY + additionY, addX + additionX])
        #             PrintSpecificMatAndPnt(occuppyMat, points)
        #             break

        #     if foundLoc:
        #         break

        # location[Y_IDX] += int(neededSize / 2)
        # location[X_IDX] += int(neededSize / 2)

        return location

    def HaveSpace(self, occuppyMat, heightsMap, yStart, xStart, neededSize):
        height = heightsMap[yStart][xStart]
        if height == 0:
            return False

        for y in range (yStart, yStart + neededSize):
            for x in range (xStart, xStart + neededSize):
                if self.bool and xStart == 29:
                    print(occuppyMat[y][x])
                if occuppyMat[y][x] or height != heightsMap[y][x]:
                    return False
        
        return True

    def CreateCameraHeightMap(self,obs):
        height_map = obs.observation['minimap'][_HEIGHT_MAP]
        cameraHeightMap = np.zeros((SCREEN_SIZE[Y_IDX], SCREEN_SIZE[X_IDX]), dtype=int)
        for y in range(0, SCREEN_SIZE[Y_IDX]):
            for x in range(0, SCREEN_SIZE[X_IDX]):
                point = [y, x]
                scaledPnt = Scale2MiniMap(point, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                cameraHeightMap[y][x] = height_map[scaledPnt[Y_IDX]][scaledPnt[X_IDX]]

        return cameraHeightMap

    def GetLocationForOilRefinery(self, obs, numRefineries):
        gasMat = self.unit_type == _VESPENE_GAS_FIELD

        vg_y, vg_x = gasMat.nonzero()
        if len(vg_y) == 0:
            return [-1, -1]
        
        if numRefineries == 0:

            location = vg_y[0], vg_x[0]
            vg_y, vg_x = IsolateArea(location, gasMat)
            midPnt = FindMiddle(vg_y, vg_x)
            return midPnt
        elif numRefineries == 1:
            rad2Include = 4
            build_y, build_x = (self.unit_type == _TERRAN_OIL_REFINERY).nonzero()

            initLoc = False
            for pnt in range(0, len(vg_y)):
                found = False
                i = 0
                while not found and i < len(build_y):
                    if abs(build_y[i] - vg_y[pnt]) < rad2Include and abs(build_x[i] - vg_x[pnt]) < rad2Include:
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
    
    def IsLastStep(self, obs):
        if self.current_state[STATE_MINERALS_IDX] < 100:
            mineralMat = self.unit_type == _NEUTRAL_MINERAL_FIELD
            for y in range (0, SCREEN_SIZE[Y_IDX]):
                for x in range (0, SCREEN_SIZE[X_IDX]):
                    if mineralMat[y][x]:
                        return False
            return True
        
        return False

    def UpdateBuildingCompletion(self, obs, buildingType):

        buildingStatus = obs.observation['multi_select']
        # print(buildingStatus)
        specificBuildingIdx = 0
        flyingBuilding = []
        for building in self.buildingVec[:]:
            if building.m_buildingType == buildingType:
                if len(buildingStatus) > specificBuildingIdx:
                    if not building.m_buildingFinished and buildingStatus[specificBuildingIdx][_BUILDING_COMPLETION_IDX] == 0:
                        building.m_buildingFinished = True
                    if buildingStatus[specificBuildingIdx][_BUILDING_TRNSPORT_SLOT] > 0:
                        flyingBuilding.append(building)
                specificBuildingIdx += 1

        # if len(buildingStatus) != specificBuildingIdx and buildingType != _TERRAN_COMMANDCENTER:
        #     print("building is not the same size for building type = ", buildingType, "num dead/flying buildings = ", specificBuildingIdx - len(buildingStatus))
        #     print("\n", buildingStatus)
        #     if buildingType == _TERRAN_BARRACKS or buildingType == _TERRAN_FACTORY:
        #         self.PrintScreen()
        #         flyingLoc = self.SearchFlyingBuilding(buildingType)
        #         if len(flyingLoc) > 0:
        #             print("flying location founded in", flyingLoc)
        #         self.SearchNewBuildingPnt()

        # if len(flyingBuilding) > 0:
        #     print("num flying buiding = ", len(flyingBuilding))
               
    def PrintMiniMap(self, obs):
        selfPnt_y, selfPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        for y in range(MINI_MAP_START[Y_IDX], MINI_MAP_END[Y_IDX]):
            for x in range(MINI_MAP_START[X_IDX], MINI_MAP_END[X_IDX]):
                isBuilding = False
                for itr in self.buildingVec[:]:
                    if x == itr.m_minimapLocation[X_IDX] and y == itr.m_minimapLocation[Y_IDX]:
                        isBuilding = True

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
                elif isBuilding:
                    print ('B', end = '')
                elif isSelf:
                    print ('s', end = '')
                elif isEnemy:
                    print ('e', end = '')
                else:
                    print ('_', end = '')
            print('|')  

    def PrintScreen(self, addPoints = []):
        buildingPoints = []
        for building in self.buildingVec[:]:
            point = Scale2Screen(building.m_minimapLocation, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
            buildingPoints.append(point)

        cc_y, cc_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()
        scv_y, scv_x = (self.unit_type == _TERRAN_SCV).nonzero()
        sd_y, sd_x = (self.unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        vg_y, vg_x = (self.unit_type == _TERRAN_OIL_REFINERY).nonzero()

        ba_y, ba_x = (self.unit_type == _TERRAN_BARRACKS).nonzero()
        fa_y, fa_x = (self.unit_type == _TERRAN_FACTORY).nonzero()
        
        re_y, re_x = (self.unit_type == _TERRAN_REACTOR).nonzero()

        fl_y, fl_x = (self.unit_type == _TERRAN_FLYING_BARRACKS).nonzero()

        mf_y, mf_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
        gas_y, gas_x = (self.unit_type == _VESPENE_GAS_FIELD).nonzero()
        
        for y in range(0, SCREEN_SIZE[Y_IDX]):
            for x in range(0, SCREEN_SIZE[X_IDX]):
                printed = False

                for i in range(0, len (addPoints)):
                    if addPoints[i][X_IDX] == x and addPoints[i][Y_IDX] == y:
                        print ('~', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (buildingPoints)):
                    if buildingPoints[i][X_IDX] == x and buildingPoints[i][Y_IDX] == y:
                        print (' ', end = '')
                        printed = True

                if printed:
                    continue
                

                for i in range(0, len (cc_y)):
                    if cc_x[i] == x and cc_y[i] == y:
                        print ('C', end = '')
                        printed = True

                if printed:
                    continue
                
                for i in range(0, len (sd_y)):
                    if sd_x[i] == x and sd_y[i] == y:
                        print ('D', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (ba_y)):
                    if ba_x[i] == x and ba_y[i] == y:
                        print ('B', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (vg_y)):
                    if vg_x[i] == x and vg_y[i] == y:
                        print ('G', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (fa_y)):
                    if fa_x[i] == x and fa_y[i] == y:
                        print ('F', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (re_y)):
                    if re_x[i] == x and re_y[i] == y:
                        print ('R', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (fl_y)):
                    if fl_x[i] == x and fl_y[i] == y:
                        print ('F', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (scv_y)):
                    if scv_x[i] == x and scv_y[i] == y:
                        print ('s', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (mf_y)):
                    if mf_x[i] == x and mf_y[i] == y:
                        print ('m', end = '')
                        printed = True

                if printed:
                    continue

                for i in range(0, len (gas_y)):
                    if gas_x[i] == x and gas_y[i] == y:
                        print ('g', end = '')
                        printed = True

                if printed:
                    continue

                print ('_', end = '')
            print('|') 

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
