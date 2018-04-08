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
_BUILD_OIL_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id

# train army action
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_CAMERA = features.MINIMAP_FEATURES.camera.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

#player general info
_MINERALS = 1
_VESPENE = 2


_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_OIL_REFINERY = 20
_TERRAN_BARRACKS = 21

_NEUTRAL_MINERAL_FIELD = 341
_VESPENE_GAS_FIELD = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

Q_TABLE_FILE = 'q_table_with_vespene'
T_TABLE_FILE = 'transition_table'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_REFINERY = 'buildsupplydepot'
ACTION_TRAIN_ARMY = 'trainarmy'
ACTION_ATTACK = 'attack'

ID_ACTION_DO_NOTHING = 0
ID_ACTION_BUILD_SUPPLY_DEPOT = 1
ID_ACTION_BUILD_BARRACKS = 2
ID_ACTION_BUILD_REFINERY = 3
ID_ACTION_TRAIN_ARMY = 4
ID_ACTION_ATTACK = 5

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_REFINERY,
    ACTION_TRAIN_ARMY,
    ACTION_ATTACK,
]

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
STATE_GRID_SIZE = 2
STATE_POWER_BUCKETING = 10
STATE_SELF_ARMY_BUCKETING = 5

STATE_COMMAND_CENTER_IDX = 0
STATE_SUPPLY_DEPOT_IDX = 1
STATE_BARRACKS_IDX = 2
STATE_REFINERY_IDX = 3
STATE_ARMY_IDX = 4
STATE_ENEMY_BASE_POWER_IDX = 5
STATE_ENEMY_ARMY_POWER_IDX = 6
STATE_ENEMY_ARMY_LOCATION_IDX = 7
STATE_SIZE = 8

CREATE_DETAILED_STATE = True
DETAILED_STATE_MINERALS_IDX = 8
DETAILED_STATE_HOTSPOTS_IDX = 9
DETAILED_STATE_SIZE = 10

class BuildingCoord:
    def __init__(self, buildingType, coordinate):
        self.m_buildingType = buildingType
        self.m_location = coordinate

class AttackCmd:
    def __init__(self, state, attackCoord, isBaseAttack ,attackStarted = False):
        self.m_state = state
        self.m_attackCoord = [math.ceil(attackCoord[Y_IDX]), math.ceil(attackCoord[X_IDX])]
        self.m_inTheWayBattle = [-1, -1]
        self.m_isBaseAttack = isBaseAttack
        self.m_attackStarted = attackStarted
        self.m_attackEnded = False

class BuildingCmd(BuildingCoord):
    def __init__(self, state, buildingType, coordinate):
        BuildingCoord.__init__(self, buildingType, coordinate)
        self.m_state = state



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

def Scale2Camera(point, camNorthWestCorner, camSouthEastCorner):
    if point[Y_IDX] < camNorthWestCorner[Y_IDX] or point[Y_IDX] > camSouthEastCorner[Y_IDX] or point[X_IDX] < camNorthWestCorner[X_IDX] or point[X_IDX] > camSouthEastCorner[X_IDX]:
        return [-1, -1]


    scaledPoint = [0,0]
    scaledPoint[Y_IDX] = point[Y_IDX] - camNorthWestCorner[Y_IDX]
    scaledPoint[X_IDX] = point[X_IDX] - camNorthWestCorner[X_IDX]  

    mapSize = [camNorthWestCorner[Y_IDX] - camSouthEastCorner[Y_IDX], camNorthWestCorner[X_IDX] - camSouthEastCorner[X_IDX]]
    
    scaledPoint[Y_IDX] = int(math.ceil(scaledPoint[Y_IDX] * SCREEN_SIZE[Y_IDX] / mapSize[Y_IDX]) - 1)
    scaledPoint[X_IDX] = int(math.ceil(scaledPoint[X_IDX] * SCREEN_SIZE[X_IDX] / mapSize[X_IDX]) - 1)

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

def PrintSpecificMat(mat, range2Include):
    for y in range(range2Include, MINIMAP_SIZE[Y_IDX] - range2Include):
        for x in range(range2Include, MINIMAP_SIZE[X_IDX] - range2Include):
            sPower = PowerSurroundPnt([y,x], range2Include, mat)
            if sPower > 9:
                print(sPower, end = ' ')
            else:
                print(sPower, end = '  ')
        print('|')

def PrintSpecificMatAndPnt(mat, range2Include, point):
    for y in range(range2Include, MINIMAP_SIZE[Y_IDX] - range2Include):
        for x in range(range2Include, MINIMAP_SIZE[X_IDX] - range2Include):
            if x == point[X_IDX] and y == point[Y_IDX]:
                print("PpP", end = '')
            else:
                sPower = PowerSurroundPnt([y,x], range2Include, mat)
                if sPower > 9:
                    print(sPower, end = ' ')
                else:
                    print(sPower, end = '  ')
        print('|')

def ScaleLoc2Grid(x, y):
    x /= (MINIMAP_SIZE[X_IDX] / STATE_GRID_SIZE)
    y /= (MINIMAP_SIZE[Y_IDX] / STATE_GRID_SIZE)
    return int(x) + int(y) * STATE_GRID_SIZE

def SwapPnt(point):
    return point[1], point[0]

def GetCoord(idxLocation, gridSize_x):
    ret = [-1,-1]
    ret[Y_IDX] = int(idxLocation / gridSize_x)
    ret[X_IDX] = idxLocation % gridSize_x
    return ret

def IsBuildAction(action):
    return action >= ID_ACTION_BUILD_SUPPLY_DEPOT and action < ID_ACTION_TRAIN_ARMY

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
        self.NumWinsTotalSlot = 1
        self.NumLossTotalSlot = 2
        
        self.NumWinsSlot = 3
        self.NumLossSlot = 4

        self.NumExpRuns = 0
        self.NumExpWins = 0
        self.NumExpLoss = 0

        self.avgScore = 0
        self.avgScoreLoss = 0
        self.avgScoreWin = 0


        self.check_state_exist(self.TrialsData)
        self.q_table.ix[self.TrialsData, self.NumWinsSlot] = 0
        self.q_table.ix[self.TrialsData, self.NumLossSlot] = 0
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
    def end_run(self, r, score):
        
        self.q_table.ix[self.TrialsData, self.NumRunsTotalSlot] += 1
        
        if (r > 0):
            self.q_table.ix[self.TrialsData, self.NumWinsTotalSlot] += 1
            self.NumExpWins += 1

            self.q_table.ix[self.TrialsData, self.NumWinsSlot] = self.NumExpWins
            self.avgScoreWin = (self.avgScoreWin * (self.NumExpWins - 1) + score) / self.NumExpWins

        elif (r < 0):
            self.q_table.ix[self.TrialsData, self.NumLossTotalSlot] += 1
            self.NumExpLoss += 1

            self.q_table.ix[self.TrialsData, self.NumLossSlot] = self.NumExpLoss
            self.avgScoreLoss = (self.avgScoreLoss * (self.NumExpLoss - 1) + score) / self.NumExpLoss
        
        self.NumExpRuns += 1
        self.avgScore = (self.avgScore * (self.NumExpRuns - 1) + score) / self.NumExpRuns

        print("curr score = " , score, "avg score: all trials = ",  self.avgScore, "losses = ", self.avgScoreLoss, ", wins = ", self.avgScoreWin)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

# class TransitionTable:
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#         self.actions = actions  # a list
#         self.tTable = pd.DataFrame(columns=self.actions, dtype=np.int)
#        # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

#     def insert(self, s, a, s_):
#         # remove start and end bracelets from states
#         s = s[:-1]
#         s_ = s_[1:]
#         allState = s + s_

#         self.check_state_exist(allState)            
#         # update
#         self.tTable.ix[allState, a] += 1

#     def check_state_exist(self, state):
#         if state not in self.tTable.index:
#             # append new state to q table
#             self.tTable = self.tTable.append(pd.Series([0] * len(self.actions), index=self.tTable.columns, name=state))

class TestAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TestAgent, self).__init__()
       
        self.qTable = QLearningTable(actions=list(range(len(smart_actions))))
        # self.tTable = TransitionTable(actions=list(range(len(smart_actions))))

        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.previous_Detailedstate = np.zeros(DETAILED_STATE_SIZE, dtype=np.int32, order='C')
        self.current_Detailedstate = np.zeros(DETAILED_STATE_SIZE, dtype=np.int32, order='C')

        self.baseDestroyed = False

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

        self.attackCommands = []

        self.topLeftBaseLocation = [23,18]
        self.bottomRightBaseLocation = [45,39]

        if os.path.isfile(Q_TABLE_FILE + '.gz'):
            self.qTable.q_table = pd.read_pickle(Q_TABLE_FILE + '.gz', compression='gzip')
        
        # if os.path.isfile(T_TABLE_FILE + '.gz'):
        #     self.tTable.tTable = pd.read_pickle(T_TABLE_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def transformLocation(self, y, x):
        if not self.base_top_left:
            return [64 - y, 64 - x]
        
        return [y, x]

    def sendToDecisionMaker(self):
        byte_array_current_state = self.current_state_for_decision_making.tobytes()
        result = -1
        data = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('172.23.40.129',5432)  #('localhost', 10000)
        message = byte_array_current_state
        try:
            # Send data
            sent = sock.sendto(message, server_address)
            # Receive response
            sock.setblocking(0)
            sock.settimeout(10)
            data, server = sock.recvfrom(5432)
            result = 0
            if(data != None):
                result = int(data[0])
                #for b in data:
                #    result = result * 256 + int(b)
        except (ConnectionResetError, socket.timeout):
            pass
        finally:
            sock.close()
          #  if(type(data) != 'bytes' or len(data) == 0):
          #      raise RuntimeError('error: wrong  data - {}  of type {}'.format(data, type(data)))
        self.returned_action_from_decision_maker = result
        return result

    def step(self, obs):
        super(TestAgent, self).step(obs)
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
            

            # if (self.step_num % 10 == 0):
                # self.PrintMiniMap(obs)
                # self.PrintScreen()
                # print("\nbuilding vector:")
                # for itr in self.buildingVec[:]:
                #     print(itr.m_buildingType, itr.m_location)
                # print("\n")

            self.ComputeAttackResults(obs)
            self.AddNewBuilding(obs)

            if self.move_number == 0:
                self.move_number += 1

                self.CreateState(obs)

                if self.previous_action is not None:
                    self.qTable.learn(str(self.previous_scaled_state), self.previous_action, 0, str(self.current_scaled_state))

                #mcts_action = self.sendToDecisionMaker(current_state.tobytes())
                mcts_action = -1
                if(self.sentToDecisionMakerAsync == None or not self.sentToDecisionMakerAsync.isAlive()):
                    if(self.returned_action_from_decision_maker != -1):
                        mcts_action =  self.returned_action_from_decision_maker
                        if(mcts_action >= 0 and mcts_action < 22):
                            print('mcts_action - {}'.format(smart_actions[mcts_action]))
                    self.returned_action_from_decision_maker = -1

                    self.sentToDecisionMakerAsync = threading.Thread(target=self.sendToDecisionMaker)

                    self.current_state_for_decision_making = self.current_state

                    self.sentToDecisionMakerAsync.start()

                rl_action = self.qTable.choose_action(str(self.current_scaled_state))

                self.previous_state[:] = self.current_state[:]
                self.previous_scaled_state[:] = self.current_scaled_state[:]
                self.previous_action = rl_action if mcts_action < 0  and mcts_action < 5 else mcts_action

                action = self.previous_action

                if IsBuildAction(action):
                    # select scv
                    unit_y, unit_x = (self.unit_type == _TERRAN_SCV).nonzero()
                        
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]
                        
                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                    
                elif action == ID_ACTION_TRAIN_ARMY:
                    barracks_y, barracks_x = (self.unit_type == _TERRAN_BARRACKS).nonzero()
                    if barracks_y.any():
                        i = random.randint(0, len(barracks_y) - 1)
                        target = [barracks_x[i], barracks_y[i]]
                        
                        return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
                    
                elif action == ID_ACTION_ATTACK:
                    if _SELECT_ARMY in obs.observation['available_actions']:
                        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            
            elif self.move_number == 1:
                self.move_number += 1
                
                action = self.previous_action
                if action == ID_ACTION_BUILD_SUPPLY_DEPOT:
                    sdCount = self.current_state[STATE_SUPPLY_DEPOT_IDX]
                    if sdCount < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                        if self.cc_y.any():
                            if sdCount == 0:
                                target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                            elif sdCount == 1:
                                target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
                            
                            coord = SwapPnt(target)
                            self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_SUPPLY_DEPOT, coord))
                            return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
                
                elif action == ID_ACTION_BUILD_BARRACKS:
                    baCount = self.current_state[STATE_BARRACKS_IDX]
                    if baCount < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                        if self.cc_y.any():
                            if  baCount == 0:
                                target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                            elif  baCount == 1:
                                target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)
                            
                            coord = SwapPnt(target)
                            self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_BARRACKS, coord))
                            return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

                elif action == ID_ACTION_BUILD_REFINERY:
                    refCount = self.current_state[STATE_REFINERY_IDX]
                    if refCount < 2 and _BUILD_OIL_REFINERY in obs.observation['available_actions']:
                        if self.cc_y.any():
                            target = self.GetLocationForOilRefinery(obs, refCount)
                            if target[0] != -1:
                                self.buildCommands.append(BuildingCmd(self.previous_state, _TERRAN_OIL_REFINERY, target))
                                return actions.FunctionCall(_BUILD_OIL_REFINERY, [_NOT_QUEUED, SwapPnt(target)])

        
                elif action == ID_ACTION_TRAIN_ARMY:

                    if _TRAIN_REAPER in obs.observation['available_actions']:
                        return actions.FunctionCall(_TRAIN_REAPER, [_QUEUED])

                    if _TRAIN_MARINE in obs.observation['available_actions']:
                        return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
            
                elif action == ID_ACTION_ATTACK:
                    do_it = True
                    if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                        do_it = False
                    
                    if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                        do_it = False

                    if len (self.attackCommands) > 0:
                        do_it = False                    
                    
                    if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                        # select target
                        isInBase = True   
                        if self.previous_state[STATE_ENEMY_ARMY_LOCATION_IDX] != STATE_NON_VALID_NUM:
                            isInBase = False
                            target = GetCoord(self.previous_state[STATE_ENEMY_ARMY_LOCATION_IDX], MINIMAP_SIZE[X_IDX])

                        elif self.baseDestroyed:
                            # find target
                            target = self.FindClosestToBaseEnemy(obs)
                            if target[Y_IDX] == -1:
                                isInBase = True
                            else:
                                isInBase = False
                        
                        if isInBase:
                            if self.base_top_left:
                                target = self.bottomRightBaseLocation
                            else:
                                target = self.topLeftBaseLocation
                        
                        self.attackCommands.append(AttackCmd(self.previous_state, target, isInBase))
                        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, SwapPnt(target)])
                    
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

        self.baseDestroyed = False
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

        cc = BuildingCoord(_TERRAN_COMMANDCENTER, miniMapLoc)
        self.buildingVec.append(cc)

    def LastStep(self, obs):
        reward = obs.reward

        if reward == 0 and not self.baseDestroyed:
            selfMat = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF)
            enemyMat = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE)

            self.PrintMiniMap(obs)
            if self.IsBaseDestroyed(selfMat, enemyMat, obs):
                print("ERROR : function returned true and in self base was not destroyed")
            else:
                if self.base_top_left:
                    enemyBaseCoord = self.bottomRightBaseLocation
                else:
                    enemyBaseCoord = self.topLeftBaseLocation
                # find largest radius of self presence

                radiusSelf = -1
                slefInEnemyBase = False
                while not slefInEnemyBase:
                    radiusSelf += 1
                    minX = max(enemyBaseCoord[X_IDX] - radiusSelf, 0)
                    maxX = max(enemyBaseCoord[X_IDX] + radiusSelf, MINIMAP_SIZE[X_IDX])

                    minY = max(enemyBaseCoord[Y_IDX] - radiusSelf, 0)
                    maxY = max(enemyBaseCoord[Y_IDX] + radiusSelf, MINIMAP_SIZE[Y_IDX])
                    if minY < 0 and minX < 0 and maxX >= MINIMAP_SIZE[X_IDX] and maxY >= MINIMAP_SIZE[Y_IDX]:
                        print("\n\nERROR : traverse all map\n\n")
                        break
                    for x in range (minX, maxX):
                        for y in range (minY, maxY):
                            if selfMat[y][x]:
                                slefInEnemyBase = True
                                break
                        if slefInEnemyBase:
                            break

                radiusEnemy = -1
                enemyPres = False
                while not enemyPres:
                    radiusEnemy += 1
                    minX = max(enemyBaseCoord[X_IDX] - radiusSelf, 0)
                    maxX = max(enemyBaseCoord[X_IDX] + radiusSelf, MINIMAP_SIZE[X_IDX])

                    minY = max(enemyBaseCoord[Y_IDX] - radiusSelf, 0)
                    maxY = max(enemyBaseCoord[Y_IDX] + radiusSelf, MINIMAP_SIZE[Y_IDX])
                    if minY < 0 and minX < 0 and maxX >= MINIMAP_SIZE[X_IDX] and maxY >= MINIMAP_SIZE[Y_IDX]:
                        print("\n\nERROR : traverse all map\n\n")
                        break
                    for x in range (minX, maxX):
                        for y in range (minY, maxY):
                            if enemyMat[y][x]:
                                enemyPres = True
                        if slefInEnemyBase:
                            break

                print("bound founded: self bound = ", radiusSelf, "enemy bound = ", radiusEnemy)


        self.qTable.learn(str(self.previous_scaled_state), self.previous_action, reward, 'terminal')
        
        score = obs.observation['score_cumulative'][0]

        self.qTable.end_run(reward, score)
        self.qTable.q_table.to_pickle(Q_TABLE_FILE + '.gz', 'gzip') 

        # if (len(self.attackCommands) > 0)

        # self.tTable.tTable.to_pickle(T_TABLE_FILE + '.gz', 'gzip') 

        self.move_number = 0
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.current_state_for_decision_making = None
        self.cc_y = None
        self.cc_x = None
        self.step_num = 0 

        self.buildingVec.clear()
        self.buildCommands.clear()
        self.attackCommands.clear()

    def CreateState(self, obs):
        commandCenterCount, supplyDepotCount, barracksCount, refineryCount = self.GetBuildingCounts()

        self.current_state[STATE_COMMAND_CENTER_IDX] = commandCenterCount
        self.current_state[STATE_SUPPLY_DEPOT_IDX] = supplyDepotCount
        self.current_state[STATE_BARRACKS_IDX] = barracksCount
        self.current_state[STATE_REFINERY_IDX] = refineryCount
        self.current_state[STATE_ARMY_IDX] = obs.observation['player'][_ARMY_SUPPLY]

        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        minDist = MAX_MINIMAP_DIST

        enemyArmyPower = 0   
        enemyBasePower = 0
        closestEnemyLocationIdx = -1
        for i in range(0, len(enemy_y)):
            [y, x] = self.transformLocation(enemy_y[i], enemy_x[i])

            if (y > MINIMAP_SIZE[Y_IDX] / 2 and x > MINIMAP_SIZE[X_IDX] / 2):
                enemyBasePower += 1
            else:

                enemyArmyPower += 1 
                y -= self.topLeftBaseLocation[Y_IDX]
                x -= self.topLeftBaseLocation[X_IDX]
                dist = x * x + y * y
                if (dist < minDist):
                    minDist = dist
                    closestEnemyLocationIdx = i
        

        if enemyArmyPower == 0:
            self.current_state[STATE_ENEMY_ARMY_POWER_IDX] = STATE_NON_VALID_NUM 
            self.current_state[STATE_ENEMY_ARMY_LOCATION_IDX] = STATE_NON_VALID_NUM
        else:
            self.current_state[STATE_ENEMY_ARMY_POWER_IDX] = enemyArmyPower
            self.current_state[STATE_ENEMY_ARMY_LOCATION_IDX] = enemy_x[closestEnemyLocationIdx] + enemy_y[closestEnemyLocationIdx] * MINIMAP_SIZE[X_IDX]
        
        if enemyBasePower == 0:
            self.current_state[STATE_ENEMY_BASE_POWER_IDX] = STATE_NON_VALID_NUM 
        else:
            self.current_state[STATE_ENEMY_BASE_POWER_IDX] = enemyBasePower            

        if CREATE_DETAILED_STATE:
            self.CreateDetailedState(obs)

        self.ScaleCurrState()

    def CreateDetailedState(self, obs):
        self.current_Detailedstate[DETAILED_STATE_MINERALS_IDX] = obs.observation['player'][_MINERALS]
        
        hot_squares = np.zeros(16)   
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16)) - 1
            x = int(math.ceil((enemy_x[i] + 1) / 16)) - 1
            idx = x + y * 4
            hot_squares[idx] = 1            
      
        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
        
        # print("hot spots:")
        # for y in range(0, 4):
        #     for x in range(0, 4):
        #         if hot_squares[x + y * 4]:
        #             print('h', end = '')
        #         else:
        #             print('_', end = '')
        #     print('|')

        val = 0
        for i in range(0, 16):
            if hot_squares[i]:
                val |= 1 << i

        self.current_Detailedstate[DETAILED_STATE_HOTSPOTS_IDX] = val


    def AddNewBuilding(self, obs):
        for itr in self.buildCommands[:]:
            buildingCoord_y, buildingCoord_x = (self.unit_type == itr.m_buildingType).nonzero()
            isBuilt = False
            for i in range(0, len(buildingCoord_y)):
                if (itr.m_location[Y_IDX] == buildingCoord_y[i] and itr.m_location[X_IDX] == buildingCoord_x[i]):
                    isBuilt = True

            if isBuilt:
                buildingMap = self.unit_type == itr.m_buildingType
                loc = [int(itr.m_location[0]),int(itr.m_location[1])]
                buildingPnt_y, buildingPnt_x = IsolateArea(loc, buildingMap)
                midPnt = FindMiddle(buildingPnt_y, buildingPnt_x)

                miniMapCoord = Scale2MiniMap(midPnt, self.cameraCornerNorthWest , self.cameraCornerSouthEast)
                isExist = False
                for itr2 in self.buildingVec[:]:
                    if itr2.m_buildingType == itr.m_buildingType and itr2.m_location == itr.m_location:
                        isExist = True

                # add building to vector if its exist
                if not isExist:    
                    self.buildingVec.append(BuildingCoord(itr.m_buildingType, miniMapCoord))
                    
                    # insert state to transition Table
                    # if itr.m_buildingType == _TERRAN_SUPPLY_DEPOT:
                    #     self.tTable.insert(str(itr.m_state), ID_ACTION_BUILD_SUPPLY_DEPOT, str(self.previous_state))

                    # elif itr.m_buildingType == _TERRAN_BARRACKS:
                    #     self.tTable.insert(str(itr.m_state), ID_ACTION_BUILD_BARRACKS, str(self.previous_state))
                
                # remove build command from vector
                
                self.buildCommands.remove(itr)
                
        self.RemoveDestroyedBuildings(obs)

    def RemoveDestroyedBuildings(self, obs):

        selfPnt_y, selfPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        # remove duplicate vals
        toRemove = []
        for curr in range(0, len(self.buildingVec)):
            isDead = True
            currBuilding = self.buildingVec[curr]
            for i in range(0, len(selfPnt_y)):
                if currBuilding.m_location[Y_IDX] == selfPnt_y[i] and currBuilding.m_location[X_IDX] == selfPnt_x[i]:
                    isDead = False

            if not isDead:
                isDup = False
                for prev in range(0, curr):
                    if self.buildingVec[prev].m_buildingType == currBuilding.m_buildingType and self.buildingVec[prev].m_location == currBuilding.m_location:
                        isDup = True
                        break


            if isDead or isDup:
                toRemove.append(curr)

        for i in range (0, len(toRemove)):
             self.buildingVec.remove(self.buildingVec[toRemove[i] - i])


    def GetBuildingCounts(self):
        ccCount = 0
        sdCount = 0
        baCount = 0
        refCount = 0
        for itr in self.buildingVec[:]:
            if itr.m_buildingType == _TERRAN_COMMANDCENTER:
                ccCount += 1
            elif itr.m_buildingType == _TERRAN_SUPPLY_DEPOT:
                sdCount += 1
            elif itr.m_buildingType == _TERRAN_BARRACKS:
                baCount += 1
            elif itr.m_buildingType == _TERRAN_OIL_REFINERY:
                refCount += 1

        return ccCount,  sdCount, baCount, refCount

    def ComputeAttackResults(self, obs):
        range2Include = 4

        selfMat = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF)
        enemyMat = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE)

        for attack in self.attackCommands[:]:
            if attack.m_inTheWayBattle[Y_IDX] == -1:
                enemyPower = PowerSurroundPnt(attack.m_attackCoord, range2Include, enemyMat)
                selfPower = PowerSurroundPnt(attack.m_attackCoord, range2Include, selfMat)
                
                if enemyPower > 0:
                    if attack.m_isBaseAttack:
                        powerIdx = STATE_ENEMY_BASE_POWER_IDX
                    else:
                        powerIdx = STATE_ENEMY_ARMY_POWER_IDX

                    attack.m_state[powerIdx] = max(attack.m_state[powerIdx], enemyPower)
                
            else:
                enemyPower = PowerSurroundPnt(attack.m_inTheWayBattle, range2Include, enemyMat)
                selfPower = PowerSurroundPnt(attack.m_inTheWayBattle, range2Include, selfMat)

            if not attack.m_attackStarted:
                isBattle, battleLocation_y, battleLocation_x = BattleStarted(selfMat, enemyMat)
                if isBattle:
                    if abs(attack.m_attackCoord[Y_IDX] - battleLocation_y) > range2Include or abs(attack.m_attackCoord[X_IDX] - battleLocation_x) > range2Include:
                        attack.m_inTheWayBattle = [battleLocation_y, battleLocation_x]

                    attack.m_attackStarted = True

            elif not attack.m_attackEnded:
                if enemyPower == 0:    
                    if attack.m_inTheWayBattle[Y_IDX] == -1:
                        attack.m_attackEnded = True
                    else:
                        attack.m_attackStarted = False
                        attack.m_inTheWayBattle = [-1,-1]

                elif selfPower == 0:
                    attack.m_attackEnded = True
            else:     
                if attack.m_isBaseAttack:
                    self.baseDestroyed = self.IsBaseDestroyed(selfMat, enemyMat, obs)
                # self.tTable.insert(str(attack.m_state), ID_ACTION_ATTACK, str(self.previous_state))
                self.attackCommands.remove(attack)

    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

        self.current_scaled_state[STATE_ARMY_IDX] = math.ceil(self.current_scaled_state[STATE_ARMY_IDX] / STATE_SELF_ARMY_BUCKETING) * STATE_SELF_ARMY_BUCKETING

        loc = self.current_state[STATE_ENEMY_ARMY_LOCATION_IDX]
        if loc != STATE_NON_VALID_NUM:
            y, x = GetCoord(loc, MINIMAP_SIZE[X_IDX])
            y, x = self.transformLocation(y, x)

            self.current_scaled_state[STATE_ENEMY_ARMY_LOCATION_IDX] = ScaleLoc2Grid(x, y)
            self.current_scaled_state[STATE_ENEMY_ARMY_POWER_IDX] = math.ceil(self.current_state[STATE_ENEMY_ARMY_POWER_IDX] / STATE_POWER_BUCKETING) * STATE_POWER_BUCKETING  
        
        if  self.current_state[STATE_ENEMY_BASE_POWER_IDX] != STATE_NON_VALID_NUM:
            self.current_scaled_state[STATE_ENEMY_BASE_POWER_IDX] = math.ceil(self.current_state[STATE_ENEMY_BASE_POWER_IDX] / STATE_POWER_BUCKETING) * STATE_POWER_BUCKETING  

    def FindClosestToBaseEnemy(self, obs):
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        if self.base_top_left:
            baseCoord = self.bottomRightBaseLocation
        else:
            baseCoord = self.topLeftBaseLocation
        
        minDist = MAX_MINIMAP_DIST
        minIdx = -1
        for i in range(0, len(enemyPnt_y)):
            diffX = abs(enemyPnt_x[i] - baseCoord[X_IDX])
            diffY = abs(enemyPnt_y[i] - baseCoord[Y_IDX])
            dist = diffX * diffX + diffY * diffY
            if (dist < minDist):
                minDist = dist
                minIdx = i

        if minIdx >= 0:
            return enemyPnt_y[minIdx], enemyPnt_x[minIdx]
        else:
            return -1,-1

    def IsBaseDestroyed(self, selfMat, enemyMat, obs, radius2SelfCheck = 6, radius2EnemyCheck = 4):

        if self.base_top_left:
            enemyBaseCoord = self.bottomRightBaseLocation
        else:
            enemyBaseCoord = self.topLeftBaseLocation

        # check if self inbase point
        slefInEnemyBase = False
        for x in range (enemyBaseCoord[X_IDX] - radius2SelfCheck, enemyBaseCoord[X_IDX] + radius2SelfCheck):
            for y in range (enemyBaseCoord[Y_IDX] - radius2SelfCheck, enemyBaseCoord[Y_IDX] + radius2SelfCheck):
                if selfMat[y][x]:
                    slefInEnemyBase = True
                    break
            if slefInEnemyBase:
                break

        # if self is not near base return false
        if not slefInEnemyBase:
            return False

        # if enemy is near base than return false
        for x in range (enemyBaseCoord[X_IDX] - radius2EnemyCheck, enemyBaseCoord[X_IDX] + radius2EnemyCheck):
            for y in range (enemyBaseCoord[Y_IDX] - radius2EnemyCheck, enemyBaseCoord[Y_IDX] + radius2EnemyCheck):
                if enemyMat[y][x]:
                    return False
        
        # if self in enemy base and enemy is not near enemy base enemy base is destroyed
        return True

    def GetLocationForOilRefinery(self, obs, numRefineries):
        gasMat = self.unit_type == _VESPENE_GAS_FIELD
        
        if numRefineries == 0:
            vg_y, vg_x = gasMat.nonzero()
            location = vg_y[0], vg_x[0]
            vg_y, vg_x = IsolateArea(location, gasMat)
            midPnt = FindMiddle(vg_y, vg_x)
            return midPnt
        elif numRefineries == 1:
            rad2Include = 4
            build_y, build_x = (self.unit_type == _TERRAN_OIL_REFINERY).nonzero()
            vg_y, vg_x = gasMat.nonzero()

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

    def PrintMiniMap(self, obs):
        selfPnt_y, selfPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        pair = [0,0]
        for pair[Y_IDX] in range(MINI_MAP_START[Y_IDX], MINI_MAP_END[Y_IDX]):
            for pair[X_IDX] in range(MINI_MAP_START[X_IDX], MINI_MAP_END[X_IDX]):
                isBuilding = False
                for itr in self.buildingVec[:]:
                    if pair == itr.m_location:
                        isBuilding = True

                isSelf = False
                for i in range (0, len(selfPnt_y)):
                    if (pair[Y_IDX] == selfPnt_y[i] and pair[X_IDX] == selfPnt_x[i]):
                        isSelf = True
                
                isEnemy = False
                for i in range (0, len(enemyPnt_y)):
                    if (pair[Y_IDX] == enemyPnt_y[i] and pair[X_IDX] == enemyPnt_x[i]):
                        isEnemy = True

                if pair == self.cameraCornerNorthWest or pair == self.cameraCornerSouthEast:
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

    def PrintScreen(self):
        cc_y, cc_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()
        scv_y, scv_x = (self.unit_type == _TERRAN_SCV).nonzero()
        sd_y, sd_x = (self.unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        ba_y, ba_x = (self.unit_type == _TERRAN_BARRACKS).nonzero()
        vg_y, vg_x = (self.unit_type == _TERRAN_OIL_REFINERY).nonzero()

        mf_y, mf_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
        gas_y, gas_x = (self.unit_type == _VESPENE_GAS_FIELD).nonzero()
        
        for y in range(0, SCREEN_SIZE[Y_IDX]):
            for x in range(0, SCREEN_SIZE[X_IDX]):
                printed = False
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
