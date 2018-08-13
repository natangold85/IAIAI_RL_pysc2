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
from utils_decisionMaker import BaseDecisionMaker

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

ACTION2STR = {}
for i in range(ACTIONS_START_IDX_SCOUT, ACTIONS_END_IDX_SCOUT):
    ACTION2STR[i] = "idx_" + str(i)

CONTROL_GROUP_ID_SCOUT = [6]

class SharedDataScout(EmptySharedData):
    def __init__(self):
        super(SharedDataScout, self).__init__()
        self.attackGroupIdx = CONTROL_GROUP_ID_SCOUT

class ScoutAgent(BaseAgent):
    def __init__(self,  sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList): 
        super(ScoutAgent, self).__init__()       

        self.sharedData = sharedData

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)

        self.SetGridSize(GRID_SIZE)

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1
          
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

    def step(self, obs, moveNum):
        super(ScoutAgent, self).step(obs)
        if obs.first():
            self.FirstStep(obs)
        
        if self.scoutInProgress:
            self.scoutInProgress = not self.IsScoutEnd(obs)

        return -1

    def FirstStep(self, obs):
        self.scoutChosen = False
        self.scoutInProgress = False
        self.lastAction = None

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
        self.lastAction = a
        return actions.FunctionCall(SC2_Actions.ATTACK_MINIMAP, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)]), True
    
    def Action2Str(self, a):
        if self.scoutInProgress:
            return "prevAction=" + ACTION2STR[self.lastAction]
        else:
            return ACTION2STR[a]