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
from utils_decisionMaker import DecisionMakerMngr
from utils_decisionMaker import BaseDecisionMaker

from utils_results import ResultFile

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints


AGENT_DIR = "BaseAttack/"

if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)

AGENT_NAME = "base_attack"

GRID_SIZE = 5

ACTION_DO_NOTHING = 0
ACTIONS_START_IDX_ATTACK = 1
ACTIONS_END_IDX_ATTACK = ACTIONS_START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
NUM_ACTIONS = ACTIONS_END_IDX_ATTACK

ACTION2STR = {}

ACTION2STR[ACTION_DO_NOTHING] = "Do_Nothing"
for a in range(ACTIONS_START_IDX_ATTACK, ACTIONS_END_IDX_ATTACK):
    ACTION2STR[a] = "BaseAttack_" + str(a - ACTIONS_START_IDX_ATTACK)

STATE_START_SELF_MAT = 0
STATE_END_SELF_MAT = STATE_START_SELF_MAT + GRID_SIZE * GRID_SIZE
STATE_START_ENEMY_MAT = STATE_END_SELF_MAT
STATE_END_ENEMY_MAT = STATE_START_ENEMY_MAT + GRID_SIZE * GRID_SIZE
STATE_TIME_LINE_IDX = STATE_END_ENEMY_MAT
STATE_SIZE = STATE_TIME_LINE_IDX + 1

TIME_LINE_BUCKETING = 25

NUM_UNIT_SCREEN_PIXELS = 0

for key,value in TerranUnit.ARMY_SPEC.items():
    if value.name == "marine":
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels

# possible types of decision maker

QTABLE = 'q'
DQN = 'dqn'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding' 

NAIVE = "naive"
USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, DQN_EMBEDDING_LOCATIONS])

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "hist"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

# table names

RUN_TYPES = {}
RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][DIRECTORY] = "baseAttack_q"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "baseAttack_q_qtable"
RUN_TYPES[QTABLE][HISTORY] = "baseAttack_q_replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "baseAttack_q_result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][DIRECTORY] = "baseAttack_dqn"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "baseAttack_dqn_DQN"
RUN_TYPES[DQN][HISTORY] = "baseAttack_dqn_replayHistory"
RUN_TYPES[DQN][RESULTS] = "baseAttack_dqn_result"

RUN_TYPES[DQN_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(STATE_SIZE, STATE_END_ENEMY_MAT, NUM_ACTIONS)
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DECISION_MAKER_NAME] = "baseAttack_dqn_Embedding_DQN"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][HISTORY] = "baseAttack_dqn_Embedding_replayHistory"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][RESULTS] = "baseAttack_dqn_Embedding_result"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][TYPE] = "naive"
RUN_TYPES[NAIVE][RESULTS] = ""

class SharedDataBaseAttack(EmptySharedData):
    def __init__(self):
        super(SharedDataBaseAttack, self).__init__()
        self.enemyBuildingMat = [0] * (GRID_SIZE * GRID_SIZE)


class BuildingUnit:
    def __init__(self, numScreenPixels, value = 1):
        self.numScreenPixels = numScreenPixels
        self.value = value

class NaiveDecisionMakerBaseAttack(BaseDecisionMaker):
    def __init__(self):
        super(NaiveDecisionMakerBaseAttack, self).__init__(AGENT_NAME)
        

    def choose_action(self, state, validActions, targetValues=False):
        buildingPnts = (state[STATE_START_ENEMY_MAT:STATE_END_ENEMY_MAT] > 0).nonzero()[0]
        selfLocs = (state[STATE_START_SELF_MAT:STATE_END_SELF_MAT] > 0).nonzero()[0]

        if len(selfLocs) == 0 or len(buildingPnts) == 0:
            return ACTION_DO_NOTHING

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

        return minIdx + ACTIONS_START_IDX_ATTACK

    def ActionsValues(self, state, validActions, target = True):
        vals = np.zeros(NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals

class BaseAttack(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):        
        super(BaseAttack, self).__init__(STATE_SIZE)

        self.sharedData = sharedData
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        self.history = self.decisionMaker.AddHistory()

        self.terminalState = np.zeros(STATE_SIZE, dtype=np.int, order='C')

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

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        if dmTypes[AGENT_NAME] == "none":
            return BaseDecisionMaker(AGENT_NAME)
            
        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerBaseAttack()
        else:
            runType = RUN_TYPES[dmTypes[AGENT_NAME]]

            # create agent dir
            directory = dmTypes["directory"] + "/" + AGENT_DIR
            if not os.path.isdir("./" + directory):
                os.makedirs("./" + directory)
            decisionMaker = DecisionMakerMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], agentName=AGENT_NAME,
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=AGENT_DIR+runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

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

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')

        self.enemyBuildingGridLoc2ScreenLoc = {}
        self.selfLocCoord = None      
    
    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)

    def Learn(self, reward, terminal):
        if self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)
            
            if self.isActionCommitted:
                self.history.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(NUM_ACTIONS):
                    self.history.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.history.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def IsDoNothingAction(self, a):
        return a == ACTION_DO_NOTHING

    def Action2Str(self, a, onlyAgent=False):
        return ACTION2STR[a]

    def Action2SC2Action(self, obs, a, moveNum):
        if SC2_Actions.STOP in obs.observation['available_actions']:
            sc2Action = SC2_Actions.STOP_SC2_ACTION
        else:
            sc2Action = SC2_Actions.DO_NOTHING_SC2_ACTION

        if self.current_action > ACTION_DO_NOTHING:   
            goTo = self.enemyBuildingGridLoc2ScreenLoc[self.current_action - ACTIONS_START_IDX_ATTACK].copy()
            if SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
        
        return sc2Action, True

    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions()
            else: 
                validActions = list(range(NUM_ACTIONS))
 
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)

        else:
            action = ACTION_DO_NOTHING

        self.current_action = action
        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        self.GetSelfLoc(obs)
        self.GetEnemyBuildingLoc(obs)
        self.current_state[STATE_TIME_LINE_IDX] = self.sharedData.numStep

        for idx in range(GRID_SIZE * GRID_SIZE):
           self.sharedData.enemyBuildingMat[idx] = self.current_state[STATE_START_ENEMY_MAT + idx]

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
                self.current_state[STATE_START_SELF_MAT + idx] += power

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
                self.current_state[STATE_START_ENEMY_MAT + idx] += enemyBuildingPower[i]
                self.enemyBuildingGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyBuildingGridLoc2ScreenLoc[idx], enemyBuildingPoints[i])
            else:
                self.current_state[STATE_START_ENEMY_MAT + idx] = enemyBuildingPower[i]
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
    
    def ValidActions(self):
        valid = [ACTION_DO_NOTHING]
        for key in self.enemyBuildingGridLoc2ScreenLoc.keys():
            valid.append(key + ACTIONS_START_IDX_ATTACK)

        return valid

    def PrintState(self):
        print("\n\nstate: timeline =", self.current_scaled_state[STATE_TIME_LINE_IDX], "last attack action =", self.lastValidAttackAction)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx = STATE_START_SELF_MAT + x + y * GRID_SIZE
                print(int(self.current_scaled_state[idx]), end = '')
            
            print(end = '  |  ')
            
            for x in range(GRID_SIZE):
                idx = STATE_START_ENEMY_MAT + x + y * GRID_SIZE
                print(int(self.current_scaled_state[idx]), end = '')

            print('||')

