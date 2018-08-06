# build base sub agent
import sys
import random
import math
import time
import os.path

import numpy as np
import pandas as pd

from pysc2.lib import actions

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from utils_decisionMaker import BaseDecisionMaker

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
ACTIONS_START_IDX_ATTACK = ACTIONS_END_IDX_SCOUT
ACTIONS_END_IDX_ATTACK = ACTIONS_START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
NUM_ACTIONS = ACTIONS_END_IDX_ATTACK


class ScoutAgent(BaseAgent):
    def __init__(self,  runArg = None, decisionMaker = None, isMultiThreaded = False, playList = None, trainList = None): 
        super(ScoutAgent, self).__init__()       

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()
            
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
                self.action2Coord[idx] = [p_y, p_x]

    def Action2SC2Action(self, obs, a, moveNum):
        if moveNum == 0:
            if a < ACTIONS_END_IDX_SCOUT:
                target = self.SelectScoutingUnit(obs)
                if SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), 
            else:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED]), False

        elif moveNum == 1:
            if a < ACTIONS_END_IDX_SCOUT:
                goTo = self.action2Coord[a - ACTIONS_START_IDX_SCOUT]
            else:
                goTo = self.action2Coord[a - ACTIONS_START_IDX_ATTACK]

            if SC2_Actions.ATTACK_MINIMAP in obs.observation['available_actions']:
                return actions.FunctionCall(SC2_Actions.ATTACK_MINIMAP, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)]), True
        
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def SelectScoutingUnit(self, obs):
        unit2Select = TerranUnit.MARINE
        unitMat = obs.observation['screen'][SC2_Params.UNIT_TYPE] == unit2Select
        p_y, p_x = SelectUnitValidPoints(unitMat)
        return [p_y, p_x]


    
