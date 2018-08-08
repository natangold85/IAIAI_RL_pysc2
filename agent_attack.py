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
from utils import EmptySharedData

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from utils_decisionMaker import BaseDecisionMaker

from agent_battle_mngr import BattleMngr
from agent_battle_mngr import SharedDataBattle

# params
from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints
from utils import SelectUnitValidPoints
from utils import FindMiddle

AGENT_DIR = "AttackAgent/"
AGENT_NAME = "attack_agent"

SUB_AGENT_ID_BATTLEMNGR = 0
SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_ID_BATTLEMNGR] = "BattleMngr"

CONTROL_GROUP_ID_ATTACK = [5]

GRID_SIZE = 2

ACTIONS_PREFORM_ATTACK = 0
ACTIONS_START_IDX_ATTACK = ACTIONS_PREFORM_ATTACK + 1
ACTIONS_END_IDX_ATTACK = ACTIONS_START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
NUM_ACTIONS = ACTIONS_END_IDX_ATTACK

class SharedDataAttack(SharedDataBattle):
    def __init__(self):
        super(SharedDataAttack, self).__init__()
        self.attackStarted = False
        self.armyInAttack = {}
        self.attackGroupIdx = CONTROL_GROUP_ID_ATTACK


class AttackAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList): 
        super(AttackAgent, self).__init__()  

        self.sharedData = sharedData

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = BaseDecisionMaker()

        self.subAgents = {}
        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)
            self.subAgents[key] = saClass(sharedData, dmTypes, saDM, isMultiThreaded, saPlayList, trainList)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()
            self.activeSubAgents = [self.subAgentPlay]
        else: 
            self.activeSubAgents = list(SUBAGENTS_NAMES.keys())
            
        self.SetGridSize(GRID_SIZE)

        self.battleMngrAction = None

    def GetDecisionMaker(self):
        return self.decisionMaker

    def step(self, obs, moveNum):
        super(AttackAgent, self).step(obs)
        
        if obs.first():
            self.FirstStep(obs)

        return -1
    
    def LastStep(self, obs, reward = 0):
        for sa in self.activeSubAgents:
            self.subAgents[sa].LastStep(obs, reward) 

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        for key, sa in self.subAgents.items():
            if sa.FindActingHeirarchi() >= 0:
                return key

        return -1

    def SetGridSize(self, gridSize):
        self.gridSize = gridSize
        
        self.action2Start = {}
        self.action2End = {}

        for y in range(self.gridSize):
            p_y = (SC2_Params.MINIMAP_SIZE * y /  gridSize)
            pEnd_y = (SC2_Params.MINIMAP_SIZE * (y + 1) /  gridSize)
            for x in range(self.gridSize):
                idx = x + y * gridSize
                p_x = (SC2_Params.MINIMAP_SIZE * x /  gridSize)
                pEnd_x = (SC2_Params.MINIMAP_SIZE * (x + 1) /  gridSize)
                
                self.action2Start[ACTIONS_START_IDX_ATTACK + idx] = [int(p_y), int(p_x)]
                self.action2End[ACTIONS_START_IDX_ATTACK + idx] = [int(pEnd_y), int(pEnd_x)]

    def FirstStep(self, obs):
        if self.playAgent:
            self.destinationCoord = [-1,-1]

    def IsDoNothingAction(self, a):
        return False

    def Action2SC2Action(self, obs, a, moveNum):
        if a == ACTIONS_PREFORM_ATTACK and self.AttackStarted():
            if self.playAgent:
                print("\n\n\nAttackAction!!\n\n")
                if moveNum == 0:
                    # select attack force
                    if SC2_Actions.SELECT_CONTROL_GROUP in obs.observation['available_actions']:       
                        return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, CONTROL_GROUP_ID_ATTACK]), False
                elif moveNum == 1:
                    # goto attack scene
                    army_y, army_x = obs.observation['minimap'][SC2_Params.SELECTED_IN_MINIMAP].nonzero()
                    if len(army_y) > 0:
                        middleArmy = FindMiddle(army_y, army_x)
                        if SC2_Actions.MOVE_CAMERA in obs.observation['available_actions'] >= 0:
                            self.attackPreformAction = True
                            return actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(middleArmy)]), False
                    else:
                        print("failed to select attack group")
                elif self.attackPreformAction:
                    sc2Action, terminal = self.subAgents[SUB_AGENT_ID_BATTLEMNGR].Action2SC2Action(obs, self.battleMngrAction, moveNum - 1)
                    self.attackPreformAction = not terminal
                    return sc2Action, False
                else:
                    # go back to base
                    print("\n\nGo Back to Base\n\n")
                    return actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(self.sharedData.commandCenterLoc[0])]), True

            else:
                return self.subAgents[SUB_AGENT_ID_BATTLEMNGR].Action2SC2Action(obs, self.battleMngrAction, moveNum - self.playAgent)

        if moveNum == 0:
            if a >= ACTIONS_START_IDX_ATTACK:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED]), False

        elif moveNum == 1:
            if a >= ACTIONS_START_IDX_ATTACK:
                self.sharedData.armyInAttack = self.ArmySelected(obs)
                if len(self.sharedData.armyInAttack) > 0:
                    if SC2_Actions.SELECT_CONTROL_GROUP in obs.observation['available_actions']:       
                        return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_SET, CONTROL_GROUP_ID_ATTACK]), False
                else:
                    self.sharedData.armyInAttack = {}


        elif moveNum == 2:
            if a >= ACTIONS_START_IDX_ATTACK:
                enemyMat = obs.observation["minimap"][SC2_Params.PLAYER_RELATIVE_MINIMAP] == SC2_Params.PLAYER_HOSTILE
                minDist = 100000
                minCoord = [-1 -1]
                coordStart = self.action2Start[a]
                for y in range(self.action2Start[a][0], self.action2End[a][0]):
                    for x in range(self.action2Start[a][1], self.action2End[a][1]):
                        if enemyMat[y][x]:
                            diffY = y - coordStart[0]
                            diffX = x - coordStart[1]
                            dist = diffY * diffY + diffX * diffX
                            if dist < minDist:
                                minDist = dist
                                minCoord = [y, x]

                if minCoord[0] < 0:
                    minCoord = coordStart

                if SC2_Actions.ATTACK_MINIMAP in obs.observation['available_actions']:
                    self.destinationCoord = minCoord
                    return actions.FunctionCall(SC2_Actions.ATTACK_MINIMAP, [SC2_Params.NOT_QUEUED, SwapPnt(minCoord)]), True
        
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def AttackStarted(self):
        if self.playAgent:
            return self.sharedData.attackStarted
        else:
            return SUB_AGENT_ID_BATTLEMNGR in self.activeSubAgents

    def ArmySelected(self, obs):
        unitCount = {}
        
        unitStatus = obs.observation['multi_select']
        
        if len(unitStatus) == 0:
            unitStatus = obs.observation['single_select']
            if len(unitStatus) == 0:
                return unitCount
        
        
        # count army
        for unit in unitStatus:
            uType = unit[SC2_Params.UNIT_TYPE_IDX]
            
            if uType in unitCount:
                unitCount[uType] += 1
            else:
                unitCount[uType] = 1

        return unitCount


    
