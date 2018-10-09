# build base sub agent
import sys
import random
import math
import time
import os.path
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from pysc2.lib import actions

from utils import BaseAgent

#sub-agents
from agent_base_attack import BaseAttack
from agent_army_attack import ArmyAttack

# shared data
from agent_base_attack import SharedDataBaseAttack
from agent_army_attack import SharedDataArmyAttack

# sc2 utils
from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay
from utils_decisionMaker import BaseDecisionMaker


from utils_results import ResultFile
from utils_results import PlotResults

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints

STEP_DURATION = 0

# possible types of play
AGENT_DIR = "BattleMngr/"
AGENT_NAME = "battle_mngr"

QTABLE = 'q'
DQN = 'dqn'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding' 
NAIVE_DECISION = 'naive'

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, DQN_EMBEDDING_LOCATIONS, NAIVE_DECISION])

GRID_SIZE = 5

ACTION_DO_NOTHING = 0
ACTION_ARMY_BATTLE = 1
ACTION_BASE_BATTLE = 2
NUM_ACTIONS = 3

SUB_AGENT_ARMY_BATTLE = ACTION_ARMY_BATTLE
SUB_AGENT_BASE_BATTLE = ACTION_BASE_BATTLE
ALL_SUB_AGENTS = [SUB_AGENT_ARMY_BATTLE, SUB_AGENT_BASE_BATTLE]

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_ARMY_BATTLE] = "ArmyAttack"
SUBAGENTS_NAMES[SUB_AGENT_BASE_BATTLE] = "BaseAttack"


ACTION2STR = {}
ACTION2STR[ACTION_DO_NOTHING] = "Do_Nothing"
ACTION2STR[ACTION_ARMY_BATTLE] = "Army_Battle"
ACTION2STR[ACTION_BASE_BATTLE] = "Base_Battle"

class STATE:
    START_SELF_MAT = 0
    END_SELF_MAT = GRID_SIZE * GRID_SIZE
    
    START_ENEMY_ARMY_MAT = END_SELF_MAT
    END_ENEMY_ARMY_MAT = START_ENEMY_ARMY_MAT + GRID_SIZE * GRID_SIZE
    
    START_ENEMY_BUILDING_MAT = END_ENEMY_ARMY_MAT
    END_ENEMY_BUILDING_MAT = START_ENEMY_BUILDING_MAT + GRID_SIZE * GRID_SIZE

    TIME_LINE_IDX = END_ENEMY_BUILDING_MAT

    SIZE = TIME_LINE_IDX + 1

    TIME_LINE_BUCKETING = 25


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
RUN_TYPES[QTABLE][DIRECTORY] = "battleMngr_qtable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "battleMngr_q_qtable"
RUN_TYPES[QTABLE][HISTORY] = "battleMngr_q_replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "battleMngr_q_result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][DIRECTORY] = "battleMngr_dqn"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "battleMngr_dqn_DQN"
RUN_TYPES[DQN][HISTORY] = "battleMngr_dqn_replayHistory"
RUN_TYPES[DQN][RESULTS] = "battleMngr_dqn_result"

RUN_TYPES[DQN_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][DIRECTORY] = "battleMngr_dqn_Embedding"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(STATE.SIZE, STATE.END_ENEMY_BUILDING_MAT, NUM_ACTIONS)
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DECISION_MAKER_NAME] = "battleMngr_dqn_Embedding_DQN"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][HISTORY] = "battleMngr_dqn_Embedding_replayHistory"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][RESULTS] = "battleMngr_dqn_Embedding_result"



class SharedDataBattle(SharedDataArmyAttack, SharedDataBaseAttack):
    def __init__(self):
        super(SharedDataBattle, self).__init__()


class NaiveDecisionMakerBattleMngr(BaseDecisionMaker):
    def __init__(self):
        super(NaiveDecisionMakerBattleMngr, self).__init__(AGENT_NAME)        
        self.startEnemyMat = GRID_SIZE * GRID_SIZE
        self.startBuildingMat = 2 * GRID_SIZE * GRID_SIZE
        self.endBuildingMat = 3 * GRID_SIZE * GRID_SIZE

        self.numActions = 3

    def choose_action(self, observation):
        if (observation[self.startEnemyMat:self.startBuildingMat] > 0).any():
            return ACTION_ARMY_BATTLE
        elif (observation[self.startBuildingMat:self.endBuildingMat] > 0).any():
            return ACTION_BASE_BATTLE
        else:
            return ACTION_DO_NOTHING

    def ActionValuesVec(self, state, target = True):
        vals = np.zeros(self.numActions,dtype = float)
        vals[self.choose_action(state)] = 1.0

        return vals



class BattleMngr(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):        
        super(BattleMngr, self).__init__(STATE.SIZE)
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList

        self.trainAgent = AGENT_NAME in trainList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        self.history = self.decisionMaker.AddHistory()

        self.sharedData = sharedData
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
            self.activeSubAgents = ALL_SUB_AGENTS


        self.current_action = None
        self.armyExist = True
        self.buildingsExist = True
        # state and actions:

        self.state_startSelfMat = 0
        self.state_startEnemyMat = GRID_SIZE * GRID_SIZE
        self.state_startBuildingMat = 2 * GRID_SIZE * GRID_SIZE
        self.state_timeLineIdx = 3 * GRID_SIZE * GRID_SIZE

        self.state_size = 3 * GRID_SIZE * GRID_SIZE + 1

        self.terminalState = np.zeros(self.state_size, dtype=np.int, order='C')
    

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        if dmTypes[AGENT_NAME] == "none":
            return BaseDecisionMaker(AGENT_NAME)

        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerBattleMngr()
        else:
            runType = RUN_TYPES[dmTypes[AGENT_NAME]]

            # create agent dir
            directory = dmTypes["directory"] + "/" + AGENT_DIR
            if not os.path.isdir("./" + directory):
                os.makedirs("./" + directory)
            decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], agentName=AGENT_NAME,  
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=AGENT_DIR+runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
        
        for sa in self.subAgents.values():
            ret = sa.GetAgentByName(name)
            if ret != None:
                return ret
            
        return None

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1

        for key, sa in self.subAgents.items():
            if sa.FindActingHeirarchi() >= 0:
                return key

        return -1

    def FirstStep(self, obs):        
        super(BattleMngr, self).FirstStep()

        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.subAgentsActions = {}
        for sa in range(NUM_ACTIONS):
            self.subAgentsActions[sa] = None
        
        for sa in SUBAGENTS_NAMES.keys():
            self.subAgentsActions[sa] = self.subAgents[sa].FirstStep(obs) 

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)
        
        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].EndRun(reward, score, stepNum)

    
    def Learn(self, reward, terminal):
        for sa in self.activeSubAgents:
            self.subAgents[sa].Learn(reward, terminal)

        if self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)

            if self.isActionCommitted:
                self.history.learn(self.previous_state, self.lastActionCommitted, reward, self.current_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(NUM_ACTIONS):
                    self.history.learn(self.previous_state, a, reward, self.terminalState, terminal)
                    self.history.learn(self.current_state, a, reward, self.terminalState, terminal)

        self.previous_state[:] = self.current_state[:]
        self.isActionCommitted = False

    def ChooseAction(self):
        for sa in self.activeSubAgents:
           self.subAgents[sa].ChooseAction()       
    
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions()
                if self.trainAgent:
                    targetValues = False
                    exploreProb = self.decisionMaker.ExploreProb()              
                else:
                    targetValues = True
                    exploreProb = self.decisionMaker.TargetExploreProb()     

                if np.random.uniform() > exploreProb:
                    valVec = self.decisionMaker.ActionValuesVec(self.current_state, targetValues)   
                    random.shuffle(validActions)
                    validVal = valVec[validActions]
                    action = validActions[validVal.argmax()]
                else:
                    action = np.random.choice(validActions) 
            else:
                action = self.decisionMaker.choose_action(self.current_state)
        else:
            action = self.subAgentPlay

        self.current_action = action
        return action

    def Action2Str(self, a, onlyAgent=False):
        if a == ACTION_DO_NOTHING or onlyAgent:
            return ACTION2STR[a]
        else:
            return ACTION2STR[a] + "-->" + self.subAgents[a].Action2Str(self.subAgentsActions[a])

    def IsDoNothingAction(self, a):
        return self.subAgents[a].IsDoNothingAction(self.subAgentsActions[a])

    def Action2SC2Action(self, obs, moveNum):
        print("\n\nBattle mngr action\n\n")
        if moveNum == 0:
            self.CreateState(obs)
            self.Learn()
            self.ChooseAction()

        self.isActionCommitted = True
        self.lastActionCommitted = self.current_action
        return self.subAgents[self.current_action].Action2SC2Action(obs, self.subAgentsActions[self.current_action], moveNum)

    def CreateState(self, obs):
        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].CreateState()

        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.GetSelfLoc(obs)
        for idx in range(GRID_SIZE * GRID_SIZE):
            self.current_state[STATE.START_ENEMY_BUILDING_MAT + idx] = self.sharedData.enemyBuildingMat[idx]
            self.current_state[STATE.START_ENEMY_ARMY_MAT + idx] = self.sharedData.enemyArmyMat[idx]

        #self.GetEnemyBuildingLoc(obs)
        self.current_state[self.state_timeLineIdx] = self.sharedData.numStep

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
                self.current_state[STATE.START_SELF_MAT + idx] += power

        if len(allArmy_y) > 0:
            self.selfLocCoord = [int(sum(allArmy_y) / len(allArmy_y)), int(sum(allArmy_x) / len(allArmy_x))]

    def GetEnemyArmyLoc(self, obs):
        playerType = obs.observation["feature_screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["feature_screen"][SC2_Params.UNIT_TYPE]

        enemyPoints = []
        enemyPower = []
        for unit, spec in TerranUnit.ARMY_SPEC.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            unitPoints, unitPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
            enemyPoints += unitPoints
            enemyPower += unitPower
        
        self.armyExist = False
        for i in range(len(enemyPoints)):
            self.armyExist = True
            idx = self.GetScaledIdx(enemyPoints[i])
            self.current_state[self.state_startEnemyMat + idx] += enemyPower[i]

    def GetEnemyBuildingLoc(self, obs):
        playerType = obs.observation["feature_screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["feature_screen"][SC2_Params.UNIT_TYPE]

        enemyBuildingPoints = []
        enemyBuildingPower = []
        for unit, spec in TerranUnit.BUILDING_SPEC.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            buildingPoints, buildingPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
            enemyBuildingPoints += buildingPoints
            enemyBuildingPower += buildingPower # * self.BuildingValues[spec.name]
        

        
        self.buildingsExist = False
        for i in range(len(enemyBuildingPoints)):
            self.buildingsExist = True
            idx = self.GetScaledIdx(enemyBuildingPoints[i])
            self.current_state[self.state_startBuildingMat + idx] += enemyBuildingPower[i]
     
       
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
        if self.armyExist:
            valid.append(ACTION_ARMY_BATTLE)
        if self.buildingsExist:
            valid.append(ACTION_BASE_BATTLE)
        
        return valid

    def PrintState(self):
        print("\nAttack action =", self.Action2Str())
        print("\n\nstate: timeline =", self.current_state[self.state_timeLineIdx])
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx = self.state_startSelfMat + x + y * GRID_SIZE
                if self.current_state[idx] < 10:
                    print(self.current_state[idx], end = '  ')
                else:
                    print(self.current_state[idx], end = ' ')
            
            print(end = '  |  ')
            
            for x in range(GRID_SIZE):
                idx = self.state_startEnemyMat + x + y * GRID_SIZE
                if self.current_state[idx] < 10:
                    print(self.current_state[idx], end = '  ')
                else:
                    print(self.current_state[idx], end = ' ')

            print(end = '  |  ')
            
            for x in range(GRID_SIZE):
                idx = self.state_startBuildingMat + x + y * GRID_SIZE
                if self.current_state[idx] < 10:
                    print(self.current_state[idx], end = '  ')
                else:
                    print(self.current_state[idx], end = ' ')

            print('||')
