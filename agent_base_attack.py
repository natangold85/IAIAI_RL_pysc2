# build base sub agent
import sys
import random
import math
import time
import os.path
import datetime

import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent
from utils import EmptySharedData

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from algo_decisionMaker import BaseNaiveDecisionMaker

from algo_decisionMaker import CreateDecisionMaker

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints


AGENT_DIR = "BaseAttack/"

if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)

AGENT_NAME = "base_attack"

GRID_SIZE = 5

class BaseAttackActions:
    DO_NOTHING = 0
    START_IDX_ATTACK = 1
    END_IDX_ATTACK = START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
    SIZE = END_IDX_ATTACK

class BaseAttackState:
    START_SELF_MAT = 0
    END_SELF_MAT = START_SELF_MAT + GRID_SIZE * GRID_SIZE
    START_ENEMY_MAT = END_SELF_MAT
    END_ENEMY_MAT = START_ENEMY_MAT + GRID_SIZE * GRID_SIZE
    TIME_LINE_IDX = END_ENEMY_MAT
    SIZE = TIME_LINE_IDX + 1

ACTION2STR = {}

ACTION2STR[BaseAttackActions.DO_NOTHING] = "Do_Nothing"
for a in range(BaseAttackActions.START_IDX_ATTACK, BaseAttackActions.END_IDX_ATTACK):
    ACTION2STR[a] = "BaseAttack_" + str(a - BaseAttackActions.START_IDX_ATTACK)

NUM_UNIT_SCREEN_PIXELS = 0

for key,value in TerranUnit.ARMY_SPEC.items():
    if value.name == "marine":
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels


class SharedDataBaseAttack(EmptySharedData):
    def __init__(self):
        super(SharedDataBaseAttack, self).__init__()
        self.enemyBuildingMat = [0] * (GRID_SIZE * GRID_SIZE)


class BuildingUnit:
    def __init__(self, numScreenPixels, value = 1):
        self.numScreenPixels = numScreenPixels
        self.value = value

class NaiveDecisionMakerBaseAttack(BaseNaiveDecisionMaker):
    def __init__(self, numTrials2Save=None, agentName="", resultFName=None, directory=None):
        super(NaiveDecisionMakerBaseAttack, self).__init__(numTrials2Save, agentName=agentName, resultFName=resultFName, directory=directory)


    def choose_action(self, state, validActions, targetValues=False):
        buildingPnts = (state[BaseAttackState.START_ENEMY_MAT:BaseAttackState.END_ENEMY_MAT] > 0).nonzero()[0]
        selfLocs = (state[BaseAttackState.START_SELF_MAT:BaseAttackState.END_SELF_MAT] > 0).nonzero()[0]

        if len(selfLocs) == 0 or len(buildingPnts) == 0:
            return BaseAttackActions.DO_NOTHING

        self_y = int(selfLocs[0] / GRID_SIZE)
        self_x = selfLocs[0] % GRID_SIZE

        minDist = 1000
        minIdx = -1

        for idx in buildingPnts:
            p_y = int(idx / GRID_SIZE)
            p_x = idx % GRID_SIZE
            diffX = p_x - self_x
            diffY = p_y - self_y

            dist = diffX * diffX + diffY * diffY
            if dist < minDist:
                minDist = dist
                minIdx = idx

        return minIdx + BaseAttackActions.START_IDX_ATTACK

    def ActionsValues(self, state, validActions, target = True):
        vals = np.zeros(BaseAttackActions.SIZE,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals

class BaseAttack(BaseAgent):
    def __init__(self, sharedData, configDict, decisionMaker, isMultiThreaded, playList, trainList, testList, dmCopy=None):        
        super(BaseAttack, self).__init__(BaseAttackState.SIZE)

        self.sharedData = sharedData
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker, _ = CreateDecisionMaker(agentName=AGENT_NAME, configDict=configDict, 
                            isMultiThreaded=isMultiThreaded, dmCopy=dmCopy, heuristicClass=NaiveDecisionMakerBaseAttack)

        self.history = self.decisionMaker.AddHistory()

        self.terminalState = np.zeros(BaseAttackState.SIZE, dtype=np.int, order='C')

        allTerranBuildings = TerranUnit.BUILDINGS + [Terran.SCV]
        
        self.buildingDetails = {}
        for unit in allTerranBuildings:
            if unit in TerranUnit.BUILDING_SPEC.keys():
                numPixels = TerranUnit.BUILDING_SPEC[unit].numScreenPixels
            elif unit == Terran.SCV:
                numPixels = TerranUnit.SCV_SPEC.numScreenPixels
            else:
                numPixels = TerranUnit.DEFAULT_BUILDING_NUM_SCREEN_PIXELS
            
            self.buildingDetails[unit] = BuildingUnit(numPixels)

        self.lastValidAttackAction = None
        self.enemyBuildingGridLoc2ScreenLoc = {}

    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
            
        return None
        
    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1
    
    def FirstStep(self, obs):
        super(BaseAttack, self).FirstStep()

        self.current_state = np.zeros(BaseAttackState.SIZE, dtype=np.int, order='C')
        self.current_scaled_state = np.zeros(BaseAttackState.SIZE, dtype=np.int, order='C')
        self.previous_scaled_state = np.zeros(BaseAttackState.SIZE, dtype=np.int, order='C')

        self.enemyBuildingGridLoc2ScreenLoc = {}
        self.selfLocCoord = None      
    
    def EndRun(self, reward, score, stepNum):
        if self.trainAgent or self.testAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

    def Learn(self, reward, terminal):
        if self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)
            
            if self.isActionCommitted:
                self.history.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(BaseAttackActions.SIZE):
                    self.history.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.history.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def IsDoNothingAction(self, a):
        return a == BaseAttackActions.DO_NOTHING

    def Action2Str(self, a, onlyAgent=False):
        return ACTION2STR[a]

    def Action2SC2Action(self, obs, a, moveNum):
        if SC2_Actions.STOP in obs.observation['available_actions']:
            sc2Action = SC2_Actions.STOP_SC2_ACTION
        else:
            sc2Action = SC2_Actions.DO_NOTHING_SC2_ACTION

        if self.current_action > BaseAttackActions.DO_NOTHING:   
            goTo = self.enemyBuildingGridLoc2ScreenLoc[self.current_action - BaseAttackActions.START_IDX_ATTACK].copy()
            if SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
        
        return sc2Action, True

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions(self.current_scaled_state)
            else: 
                validActions = list(range(BaseAttackActions.SIZE))
 
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)

        else:
            action = BaseAttackActions.DO_NOTHING

        self.current_action = action
        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(BaseAttackState.SIZE, dtype=np.int, order='C')
        self.GetSelfLoc(obs)
        self.GetEnemyBuildingLoc(obs)
        self.current_state[BaseAttackState.TIME_LINE_IDX] = self.sharedData.numStep

        for idx in range(GRID_SIZE * GRID_SIZE):
           self.sharedData.enemyBuildingMat[idx] = self.current_state[BaseAttackState.START_ENEMY_MAT + idx]

        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]

    def GetSelfLoc(self, obs):
        playerType = obs.observation["feature_screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["feature_screen"][SC2_Params.UNIT_TYPE]

        allArmy_y = []
        allArmy_x = [] 
        for key, spec in TerranUnit.ARMY_SPEC.items():
            s_y, s_x = ((playerType == SC2_Params.PLAYER_SELF) &(unitType == key)).nonzero()
            allArmy_y += list(s_y)
            allArmy_x += list(s_x)
            
            selfPoints, selfPower = CenterPoints(s_y, s_x)
            
            for i in range(len(selfPoints)):
                idx = self.GetScaledIdx(selfPoints[i])
                power = math.ceil(selfPower[i] / spec.numScreenPixels)
                self.current_state[BaseAttackState.START_SELF_MAT + idx] += power

        if len(allArmy_y) > 0:
            self.selfLocCoord = [int(sum(allArmy_y) / len(allArmy_y)), int(sum(allArmy_x) / len(allArmy_x))]

    def GetEnemyBuildingLoc(self, obs):
        playerType = obs.observation["feature_screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["feature_screen"][SC2_Params.UNIT_TYPE]

        enemyBuildingPoints = []
        enemyBuildingPower = []
        for unit , spec in self.buildingDetails.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            if len(enemyArmy_y) > 0:
                buildingPoints, buildingPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
                enemyBuildingPoints += buildingPoints
                enemyBuildingPower += buildingPower * spec.value
        
        self.enemyBuildingGridLoc2ScreenLoc = {}
        for i in range(len(enemyBuildingPoints)):
            idx = self.GetScaledIdx(enemyBuildingPoints[i])
            if idx in self.enemyBuildingGridLoc2ScreenLoc.keys():
                self.current_state[BaseAttackState.START_ENEMY_MAT + idx] += enemyBuildingPower[i]
                self.enemyBuildingGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyBuildingGridLoc2ScreenLoc[idx], enemyBuildingPoints[i])
            else:
                self.current_state[BaseAttackState.START_ENEMY_MAT + idx] = enemyBuildingPower[i]
                self.enemyBuildingGridLoc2ScreenLoc[idx] = enemyBuildingPoints[i]   

    def GetScaledIdx(self, screenCord):
        locX = screenCord[SC2_Params.X_IDX]
        locY = screenCord[SC2_Params.Y_IDX]

        yScaled = int((locY / SC2_Params.SCREEN_SIZE) * GRID_SIZE)
        xScaled = int((locX / SC2_Params.SCREEN_SIZE) * GRID_SIZE)

        return xScaled + yScaled * GRID_SIZE
    
    def Closest2Self(self, p1, p2):
        d1 = DistForCmp(p1, self.selfLocCoord)
        d2 = DistForCmp(p2, self.selfLocCoord)
        if d1 < d2:
            return p1
        else:
            return p2
    
    def ValidActions(self, state):
        valid = [BaseAttackActions.DO_NOTHING]
        valid = [BaseAttackActions.DO_NOTHING]
        enemiesLoc = (state[BaseAttackState.START_ENEMY_MAT:BaseAttackState.END_ENEMY_MAT] > 0).nonzero()
        for loc in enemiesLoc[0]:
            valid.append(loc + BaseAttackActions.START_IDX_ATTACK)

        return valid

    def PrintState(self):
        print("\n\nstate: timeline =", self.current_scaled_state[BaseAttackState.TIME_LINE_IDX], "last attack action =", self.lastValidAttackAction)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx = BaseAttackState.START_SELF_MAT + x + y * GRID_SIZE
                print(int(self.current_scaled_state[idx]), end = '')
            
            print(end = '  |  ')
            
            for x in range(GRID_SIZE):
                idx = BaseAttackState.START_ENEMY_MAT + x + y * GRID_SIZE
                print(int(self.current_scaled_state[idx]), end = '')

            print('||')

