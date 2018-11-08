# build base sub agent
import sys
import random
import math
import time
import os.path

import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from algo_decisionMaker import BaseDecisionMaker

from utils import EmptySharedData
# params

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints
from utils import SelectUnitValidPoints

AGENT_DIR = "ScoutAgent/"
if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)
AGENT_NAME = "scout_agent"

GRID_SIZE = 2

ACTIONS_START_IDX_SCOUT = 0
ACTIONS_END_IDX_SCOUT = ACTIONS_START_IDX_SCOUT + GRID_SIZE * GRID_SIZE
NUM_ACTIONS = ACTIONS_END_IDX_SCOUT

RANGE_TO_END_SCOUT = 7

MAX_FOG_COUNTER_2_ZERO = 40
FOG_GROUPING = 5
NON_VALID_MINIMAP_HEIGHT = 0  

ACTION2STR = {}
for i in range(ACTIONS_START_IDX_SCOUT, ACTIONS_END_IDX_SCOUT):
    ACTION2STR[i] = "idx_" + str(i)

CONTROL_GROUP_ID_SCOUT = [6]

class SharedDataScout(EmptySharedData):
    def __init__(self):
        super(SharedDataScout, self).__init__()
        self.fogRatioMat = np.zeros((GRID_SIZE, GRID_SIZE), float)
        self.fogCounterMat = np.zeros((GRID_SIZE, GRID_SIZE), int)
        self.enemyMatObservation = np.zeros((GRID_SIZE,GRID_SIZE), int)

        self.scoutGroupIdx = CONTROL_GROUP_ID_SCOUT

class ScoutAgent(BaseAgent):
    def __init__(self,  sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList, dmCopy=None): 
        super(ScoutAgent, self).__init__()       

        self.sharedData = sharedData

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)

        self.SetGridSize(GRID_SIZE)

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
            
        return None
             
    def SetGridSize(self, gridSize):
        self.gridSize = gridSize
        
        self.action2Coord = {}
        toMiddle = SC2_Params.MINIMAP_SIZE / (2 * gridSize)

        for y in range(self.gridSize):
            p_y = (SC2_Params.MINIMAP_SIZE * y /  gridSize) + toMiddle
            for x in range(self.gridSize):
                idx = x + y * gridSize
                p_x = (SC2_Params.MINIMAP_SIZE * x /  gridSize) + toMiddle
                self.action2Coord[idx] = [int(p_y), int(p_x)]

    def FirstStep(self, obs):
        super(ScoutAgent, self).FirstStep()  

        self.scoutChosen = False
        self.scoutInProgress = False

        self.fogMatFull = np.zeros((SC2_Params.MINIMAP_SIZE, SC2_Params.MINIMAP_SIZE), int)
        self.fogCounterMatFull = np.zeros((SC2_Params.MINIMAP_SIZE, SC2_Params.MINIMAP_SIZE), int)
        self.enemyMatObservationFull = np.zeros((SC2_Params.MINIMAP_SIZE,SC2_Params.MINIMAP_SIZE), bool)

        nonVal_y, nonVal_x = (obs.observation['feature_minimap'][SC2_Params.HEIGHT_MAP] == NON_VALID_MINIMAP_HEIGHT).nonzero()
        self.fogMatFull[nonVal_y,nonVal_x] = -1  
        self.fogCounterMatFull[nonVal_y,nonVal_x] = -1  
               

    def Action2SC2Action(self, obs, a, moveNum):     
        if moveNum == 0:
            if a < ACTIONS_END_IDX_SCOUT:
                if self.scoutChosen:
                    return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, CONTROL_GROUP_ID_SCOUT]), False
                else:
                    target = self.SelectScoutingUnit(obs)
                    if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                        return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), False


        elif moveNum == 1:
            if a < ACTIONS_END_IDX_SCOUT:
                if self.scoutChosen:
                    if not self.IsScouterSelected(obs):
                        self.scoutChosen = False
                    elif SC2_Actions.ATTACK_MINIMAP in obs.observation['available_actions']:
                        return self.GoToLocationAction(a)

                elif SC2_Actions.SELECT_CONTROL_GROUP in obs.observation['available_actions']:       
                    self.scoutChosen = True
                    return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_SET, CONTROL_GROUP_ID_SCOUT]), False
        
        elif moveNum == 2:
            if a < ACTIONS_END_IDX_SCOUT:
                if SC2_Actions.ATTACK_MINIMAP in obs.observation['available_actions']:
                    return self.GoToLocationAction(a)
        
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def MonitorObservation(self, obs):
        miniMapVisi = obs.observation['feature_minimap'][SC2_Params.VISIBILITY_MINIMAP]
        miniMapHeights = obs.observation['feature_minimap'][SC2_Params.HEIGHT_MAP] != NON_VALID_MINIMAP_HEIGHT
        
        # counter above threshold are treated as non-visibles
        self.fogMatFull[(miniMapHeights) & (miniMapVisi == SC2_Params.SLIGHT_FOG) & (self.fogCounterMatFull >= MAX_FOG_COUNTER_2_ZERO)] = SC2_Params.FOG

        # idx for slight fog = (valid & visi=slight & counter < threshold)
        ySlightFog, xSlightFog = ((miniMapHeights) & (miniMapVisi == SC2_Params.SLIGHT_FOG) & (self.fogCounterMatFull < MAX_FOG_COUNTER_2_ZERO)).nonzero()
        self.fogMatFull[ySlightFog, xSlightFog] = SC2_Params.SLIGHT_FOG
        self.fogCounterMatFull[ySlightFog,xSlightFog] += 1

        # treat absolute values (fog or in sight)
        yInSight, xInSight = ((miniMapHeights) & (miniMapVisi == SC2_Params.IN_SIGHT)).nonzero()
        self.fogMatFull[yInSight, xInSight] = SC2_Params.IN_SIGHT
        self.fogCounterMatFull[yInSight, xInSight] = 0  

        yFog, xFog = ((miniMapHeights) & (miniMapVisi == SC2_Params.FOG)).nonzero()
        self.fogMatFull[yFog, xFog] = SC2_Params.FOG
        self.fogCounterMatFull[yFog, xFog] = MAX_FOG_COUNTER_2_ZERO 

        # calculate enemy mat
        miniMapEnemy = obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] == SC2_Params.PLAYER_HOSTILE        
        # if enemy and in sight insert enemy
        self.enemyMatObservationFull[((miniMapEnemy) & (self.fogMatFull == SC2_Params.IN_SIGHT)).nonzero()] = True
        # if fog treat as unknown
        self.enemyMatObservationFull[(self.fogMatFull == SC2_Params.FOG).nonzero()] = False

        if self.scoutInProgress:
            self.scoutInProgress = not self.IsScoutEnd(obs)    

    def CreateState(self, obs):
        # bucket fog counter (treat fog values as max counter value)
        bucketFogCounter = (self.fogCounterMatFull / FOG_GROUPING).astype(int)

        # scale fogmat, counter and enemy observation to gridsize
        for y in range(GRID_SIZE):
            startY = int(y * (SC2_Params.MINIMAP_SIZE / GRID_SIZE))
            endY = int((y + 1) * (SC2_Params.MINIMAP_SIZE / GRID_SIZE))
            for x in range(GRID_SIZE):
                startX = int(x * (SC2_Params.MINIMAP_SIZE / GRID_SIZE))
                endX = int((x + 1) * (SC2_Params.MINIMAP_SIZE / GRID_SIZE))

                # count enemy power
                numEnemyPixels = np.sum(self.enemyMatObservationFull[startY:endY, startX:endX])
                self.sharedData.enemyMatObservation[y, x] = numEnemyPixels
                
                # calc fog ratio
                numInSight = np.sum(self.fogMatFull[startY:endY, startX:endX] > SC2_Params.FOG)
                totNum = np.sum(self.fogMatFull[startY:endY, startX:endX] >= SC2_Params.FOG)
                self.sharedData.fogRatioMat[y, x] = numInSight / totNum

                # calc most frequent fog group count
                values,counts = np.unique(bucketFogCounter[startY:endY, startX:endX],return_counts=True)
                # pop non valid values if exist
                if values[0] == -1:
                    counts.pop(0)
                    values.pop(0)
                
                self.sharedData.fogCounterMat[y, x] = values[np.argmax(counts)]

    def SelectScoutingUnit(self, obs):
        unit2Select = Terran.Marine
        unitMat = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE] == unit2Select
        p_y, p_x = SelectUnitValidPoints(unitMat)
        if len(p_y) > 0:
            return [p_y[0], p_x[0]]
        else: 
            return [-1,-1]
    
    def IsScouterSelected(self, obs):
        unitStatus = obs.observation['single_select']
        return len(unitStatus) > 0

    def IsScoutEnd(self, obs):
        selfMat = obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] == SC2_Params.PLAYER_SELF
        yTarget = self.goToLast[SC2_Params.Y_IDX]
        xTarget = self.goToLast[SC2_Params.X_IDX]
        
        # for y in range(yTarget - RANGE_TO_END_SCOUT, yTarget + RANGE_TO_END_SCOUT):
        #     for x in range(xTarget - RANGE_TO_END_SCOUT, xTarget + RANGE_TO_END_SCOUT):
        #         if selfMat[y][x]:
        #             print("\n\n\nScout Succeed!!\n\n")
        #             return True
        s_y, s_x = selfMat.nonzero()
        minDist = 1000
        
        for i in range(len(s_y)):
            diffY = s_y[i] - yTarget
            diffX = s_x[i] - xTarget
            dist = diffY * diffY + diffX * diffX
            if dist < minDist:
                minDist = dist

        realDist = math.sqrt(minDist)
        
        if realDist < RANGE_TO_END_SCOUT:
            return True
        else:
            return False

    def IsDoNothingAction(self, a):
        return False

    def GoToLocationAction(self, a):
        goTo = self.action2Coord[a - ACTIONS_START_IDX_SCOUT]
        self.scoutInProgress = True
        self.goToLast = goTo
        return actions.FunctionCall(SC2_Actions.ATTACK_MINIMAP, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)]), True
    
    def Action2Str(self, a, onlyAgent=False):
        return ACTION2STR[a]