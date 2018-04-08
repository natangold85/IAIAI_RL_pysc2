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

class SC2:
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

    Y_IDX = 0
    X_IDX = 1

    MINIMAP_SIZE = [64, 64]
    MINI_MAP_START = [8, 8]
    MINI_MAP_END = [61, 50]
    MAX_MINIMAP_DIST = MINIMAP_SIZE[X_IDX] * MINIMAP_SIZE[X_IDX] + MINIMAP_SIZE[Y_IDX] * MINIMAP_SIZE[Y_IDX] 

    SCREEN_SIZE = [84, 84]


class BuildingCoord:
    def __init__(self, buildingType, coordinate):
        self.m_buildingType = buildingType
        self.m_location = coordinate

class AttackCmd:
    def __init__(self, state, attackCoord, isBaseAttack ,attackStarted = False):
        self.m_state = state
        self.m_attackCoord = [math.ceil(attackCoord[SC2.Y_IDX]), math.ceil(attackCoord[SC2.X_IDX])]
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

    return [midd_y, midd_x]

def Flood(location, buildingMap):   
    closeLocs = [[location[SC2.Y_IDX] + 1, location[SC2.X_IDX]], [location[SC2.Y_IDX] - 1, location[SC2.X_IDX]], [location[SC2.Y_IDX], location[SC2.X_IDX] + 1], [location[SC2.Y_IDX], location[SC2.X_IDX] - 1] ]
    points_y = [location[SC2.Y_IDX]]
    points_x = [location[SC2.X_IDX]]
    for loc in closeLocs[:]:
        if buildingMap[loc[SC2.Y_IDX]][loc[SC2.X_IDX]]:
            buildingMap[loc[SC2.Y_IDX]][loc[SC2.X_IDX]] = False
            pnts_y, pnts_x = Flood(loc, buildingMap)
            points_x.extend(pnts_x)
            points_y.extend(pnts_y)  

    return points_y, points_x


def IsolateBuilding(location, buildingMap):           
    return Flood(location, buildingMap)

def Scale2MiniMap(point, camNorthWestCorner, camSouthEastCorner):
    scaledPoint = [0,0]
    scaledPoint[SC2.Y_IDX] = point[SC2.Y_IDX] * (camSouthEastCorner[SC2.Y_IDX] - camNorthWestCorner[SC2.Y_IDX]) / SC2.SCREEN_SIZE[SC2.Y_IDX] 
    scaledPoint[SC2.X_IDX] = point[SC2.X_IDX] * (camSouthEastCorner[SC2.X_IDX] - camNorthWestCorner[SC2.X_IDX]) / SC2.SCREEN_SIZE[SC2.X_IDX] 
    
    scaledPoint[SC2.Y_IDX] += camNorthWestCorner[SC2.Y_IDX]
    scaledPoint[SC2.X_IDX] += camNorthWestCorner[SC2.X_IDX]
    
    scaledPoint[SC2.Y_IDX] = math.ceil(scaledPoint[SC2.Y_IDX])
    scaledPoint[SC2.X_IDX] = math.ceil(scaledPoint[SC2.X_IDX])

    return scaledPoint

def Scale2Camera(point, camNorthWestCorner, camSouthEastCorner):
    if point[SC2.Y_IDX] < camNorthWestCorner[SC2.Y_IDX] or point[SC2.Y_IDX] > camSouthEastCorner[SC2.Y_IDX] or point[SC2.X_IDX] < camNorthWestCorner[SC2.X_IDX] or point[SC2.X_IDX] > camSouthEastCorner[SC2.X_IDX]:
        return [-1, -1]


    scaledPoint = [0,0]
    scaledPoint[SC2.Y_IDX] = point[SC2.Y_IDX] - camNorthWestCorner[SC2.Y_IDX]
    scaledPoint[SC2.X_IDX] = point[SC2.X_IDX] - camNorthWestCorner[SC2.X_IDX]  

    mapSize = [camNorthWestCorner[SC2.Y_IDX] - camSouthEastCorner[SC2.Y_IDX], camNorthWestCorner[SC2.X_IDX] - camSouthEastCorner[SC2.X_IDX]]
    
    scaledPoint[SC2.Y_IDX] = int(math.ceil(scaledPoint[SC2.Y_IDX] * SC2.SCREEN_SIZE[SC2.Y_IDX] / mapSize[SC2.Y_IDX]) - 1)
    scaledPoint[SC2.X_IDX] = int(math.ceil(scaledPoint[SC2.X_IDX] * SC2.SCREEN_SIZE[SC2.X_IDX] / mapSize[SC2.X_IDX]) - 1)

    return scaledPoint

def GetScreenCorners(obs):
    cameraLoc = obs.observation['minimap'][SC2._CAMERA]
    ca_y, ca_x = cameraLoc.nonzero()

    return [ca_y.min(), ca_x.min()] , [ca_y.max(), ca_x.max()]

def PowerSurroundPnt(point, radius2Include, powerMat):
    power = 0
    for y in range(-radius2Include, radius2Include):
        for x in range(-radius2Include, radius2Include):
            power += powerMat[y + point[SC2.Y_IDX]][x + point[SC2.X_IDX]]

    return power
def BattleStarted(selfMat, enemyMat):
    attackRange = 1

    for xEnemy in range (attackRange, SC2.MINIMAP_SIZE[SC2.X_IDX] - attackRange):
        for yEnemy in range (attackRange, SC2.MINIMAP_SIZE[SC2.Y_IDX] - attackRange):
            if enemyMat[yEnemy,xEnemy]:
                for xSelf in range(xEnemy - attackRange, xEnemy + attackRange):
                    for ySelf in range(yEnemy - attackRange, yEnemy + attackRange):
                        if enemyMat[ySelf][xSelf]:
                            return True, yEnemy, xEnemy

    return False, -1, -1

def PrintSpecificMat(mat, range2Include):
    for y in range(range2Include, SC2.MINIMAP_SIZE[SC2.Y_IDX] - range2Include):
        for x in range(range2Include, SC2.MINIMAP_SIZE[SC2.X_IDX] - range2Include):
            sPower = PowerSurroundPnt([y,x], range2Include, mat)
            if sPower > 9:
                print(sPower, end = ' ')
            else:
                print(sPower, end = '  ')
        print('|')

def PrintSpecificMatAndPnt(mat, range2Include, point):
    for y in range(range2Include, SC2.MINIMAP_SIZE[SC2.Y_IDX] - range2Include):
        for x in range(range2Include, SC2.MINIMAP_SIZE[SC2.X_IDX] - range2Include):
            if x == point[SC2.X_IDX] and y == point[SC2.Y_IDX]:
                print("PpP", end = '')
            else:
                sPower = PowerSurroundPnt([y,x], range2Include, mat)
                if sPower > 9:
                    print(sPower, end = ' ')
                else:
                    print(sPower, end = '  ')
        print('|')

def ScaleLoc2Grid(x, y, gridSize):
    x /= (SC2.MINIMAP_SIZE[SC2.X_IDX] / gridSize)
    y /= (SC2.MINIMAP_SIZE[SC2.Y_IDX] / gridSize)
    return int(x) + int(y) * gridSize

def SwapPnt(point):
    return point[1], point[0]

def GetCoord(idxLocation, gridSize_x):
    ret = [-1,-1]
    ret[SC2.Y_IDX] = int(idxLocation / gridSize_x)
    ret[SC2.X_IDX] = idxLocation % gridSize_x
    return ret

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

class TransitionTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.tTable = pd.DataFrame(columns=self.actions, dtype=np.int)
       # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def insert(self, s, a, s_):
        # remove start and end bracelets from states
        s = s[:-1]
        s_ = s_[1:]
        allState = s + s_

        self.check_state_exist(allState)            
        # update
        self.tTable.ix[allState, a] += 1

    def check_state_exist(self, state):
        if state not in self.tTable.index:
            # append new state to q table
            self.tTable = self.tTable.append(pd.Series([0] * len(self.actions), index=self.tTable.columns, name=state))

class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()

        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]

        self.topLeftBaseLocation = [23,18]
        self.bottomRightBaseLocation = [45,39]

    def step(self,obs):
        super(Agent, self).step(obs)

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def getBaseLocation(self):
        if self.base_top_left:
            return self.topLeftBaseLocation
        else:
            return self.bottomRightBaseLocation

    def getEnemyBaseLocation(self):
        if self.base_top_left:
            return self.bottomRightBaseLocation
        else:
            return self.topLeftBaseLocation

    def transformLocation(self, y, x):
        if not self.base_top_left:
            return [SC2.MINIMAP_SIZE[SC2.Y_IDX] - y, SC2.MINIMAP_SIZE[SC2.X_IDX] - x]
        
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

    def FirstStep(self, obs):
        player_y, player_x = (obs.observation['minimap'][SC2._PLAYER_RELATIVE] == SC2._PLAYER_SELF).nonzero()

        if player_y.any() and player_y.mean() <= 31:
            self.base_top_left = True  
        else:
            self.base_top_left = False

        self.sentToDecisionMakerAsync = None 
        self.returned_action_from_decision_maker = -1

    def LastStep(self, obs):              
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.current_state_for_decision_making = None

    def FindClosestToEnemyBase(self, obs):
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][SC2._PLAYER_RELATIVE] == SC2._PLAYER_HOSTILE).nonzero()
        baseCoord = self.getEnemyBaseLocation()
        
        minDist = SC2.MAX_MINIMAP_DIST
        minIdx = -1
        for i in range(0, len(enemyPnt_y)):
            diffX = abs(enemyPnt_x[i] - baseCoord[SC2.X_IDX])
            diffY = abs(enemyPnt_y[i] - baseCoord[SC2.Y_IDX])
            dist = diffX * diffX + diffY * diffY
            if (dist < minDist):
                minDist = dist
                minIdx = i

        if minIdx >= 0:
            return enemyPnt_y[minIdx], enemyPnt_x[minIdx]
        else:
            return -1,-1

    def IsBaseDestroyed(self, selfMat, enemyMat, obs, radius2SelfCheck = 3, radius2EnemyCheck = 5):

        if self.base_top_left:
            enemyBaseCoord = self.bottomRightBaseLocation
        else:
            enemyBaseCoord = self.topLeftBaseLocation

        # check if self inbase point
        slefInEnemyBase = False
        for x in range (enemyBaseCoord[SC2.X_IDX] - radius2SelfCheck, enemyBaseCoord[SC2.X_IDX] + radius2SelfCheck):
            for y in range (enemyBaseCoord[SC2.Y_IDX] - radius2SelfCheck, enemyBaseCoord[SC2.Y_IDX] + radius2SelfCheck):
                if selfMat[y][x]:
                    slefInEnemyBase = True
                    break
            if slefInEnemyBase:
                break

        if not slefInEnemyBase:
            return False

        # if enemy is near base than return false
        for x in range (enemyBaseCoord[SC2.X_IDX] - radius2EnemyCheck, enemyBaseCoord[SC2.X_IDX] + radius2EnemyCheck):
            for y in range (enemyBaseCoord[SC2.Y_IDX] - radius2EnemyCheck, enemyBaseCoord[SC2.Y_IDX] + radius2EnemyCheck):
                if enemyMat[y][x]:
                    return False
        
        # if self in enemy base and enemy is not near enemy base enemy base is destroyed
        return True

    def PrintMiniMap(self, obs, buildingVec = []):
        selfPnt_y, selfPnt_x = (obs.observation['minimap'][SC2._PLAYER_RELATIVE] == SC2._PLAYER_SELF).nonzero()
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][SC2._PLAYER_RELATIVE] == SC2._PLAYER_HOSTILE).nonzero()

        pair = [0,0]
        for pair[SC2.Y_IDX] in range(SC2.MINI_MAP_START[SC2.Y_IDX], SC2.MINI_MAP_END[SC2.Y_IDX]):
            for pair[SC2.X_IDX] in range(SC2.MINI_MAP_START[SC2.X_IDX], SC2.MINI_MAP_END[SC2.X_IDX]):
                isBuilding = False
                for itr in buildingVec[:]:
                    if pair == itr.m_location:
                        isBuilding = True

                isSelf = False
                for i in range (0, len(selfPnt_y)):
                    if (pair[SC2.Y_IDX] == selfPnt_y[i] and pair[SC2.X_IDX] == selfPnt_x[i]):
                        isSelf = True
                
                isEnemy = False
                for i in range (0, len(enemyPnt_y)):
                    if (pair[SC2.Y_IDX] == enemyPnt_y[i] and pair[SC2.X_IDX] == enemyPnt_x[i]):
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

    def PrintScreen(self, unit_type):
        cc_y, cc_x = (unit_type == SC2._TERRAN_COMMANDCENTER).nonzero()
        scv_y, scv_x = (unit_type == SC2._TERRAN_SCV).nonzero()
        sd_y, sd_x = (unit_type == SC2._TERRAN_SUPPLY_DEPOT).nonzero()
        ba_y, ba_x = (unit_type == SC2._TERRAN_BARRACKS).nonzero()
        mf_y, mf_x = (unit_type == SC2._NEUTRAL_MINERAL_FIELD).nonzero()
        
        for y in range(0, SC2.SCREEN_SIZE[SC2.Y_IDX]):
            for x in range(0, SC2.SCREEN_SIZE[SC2.X_IDX]):
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

                print ('_', end = '')
            print('|') 
