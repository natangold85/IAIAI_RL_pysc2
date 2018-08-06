import random
import math
import os.path
import logging
import traceback
import datetime

import numpy as np
import pandas as pd
import time
import sys

from pysc2.lib import actions

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_decisionMaker import LearnWithReplayMngr

from utils_qtable import QTableParamsExplorationDecay
from utils_dqn import DQN_PARAMS

# sub agents

from agent_do_nothing import DoNothingSubAgent
from agent_base_mngr import BaseMngr
from agent_scout import ScoutAgent
from agent_battle_mngr import BattleMngr

# shared data
from agent_base_mngr import SharedDataBase
from agent_battle_mngr import SharedDataBattle

from agent_base_mngr import BASE_STATE

AGENT_DIR = "SuperAgent/"
if not os.path.isdir("./" + AGENT_DIR):
    os.makedirs("./" + AGENT_DIR)

AGENT_NAME = "super"

STEP_DURATION = 0

SUB_AGENT_ID_DONOTHING = 0
SUB_AGENT_ID_BASE = 1
SUB_AGENT_ID_ATTACK = 2
SUB_AGENT_ID_SCOUT = 3
NUM_SUB_AGENTS = 4

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_ID_DONOTHING] = "DoNothingSubAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_BASE] = "BaseMngr"
SUBAGENTS_NAMES[SUB_AGENT_ID_SCOUT] = "ScoutAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_ATTACK] = "BattleMngr"

SUBAGENTS_ARGS = {}
SUBAGENTS_ARGS[SUB_AGENT_ID_DONOTHING] = "naive"
SUBAGENTS_ARGS[SUB_AGENT_ID_BASE] = "inherit"
SUBAGENTS_ARGS[SUB_AGENT_ID_SCOUT] = "naive"
SUBAGENTS_ARGS[SUB_AGENT_ID_ATTACK] = "naive"

# possible types of play

QTABLE = 'q'
DQN = 'dqn'
USER_PLAY = 'play'

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "hist"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN])

# state details
class STATE:

    BASE = BASE_STATE()

    GRID_SIZE = 2

    SELF_POWER_BUCKETING = 5
    ENEMY_POWER_BUCKETING = 5

    
    BASE_START = 0
    BASE_END = BASE.SIZE

    # power and fog mat
    SELF_MAT_START = BASE_END
    SELF_MAT_END = SELF_MAT_START + GRID_SIZE * GRID_SIZE
    FOG_MAT_START = SELF_MAT_END
    FOG_MAT_END = FOG_MAT_START + GRID_SIZE * GRID_SIZE
    ENEMY_ARMY_MAT_START = FOG_MAT_END
    ENEMY_ARMY_MAT_END = ENEMY_ARMY_MAT_START + GRID_SIZE * GRID_SIZE
    ENEMY_BUILDING_MAT_START = ENEMY_ARMY_MAT_END
    ENEMY_BUILDING_MAT_END = ENEMY_BUILDING_MAT_START + GRID_SIZE * GRID_SIZE  

    NON_VALID_MINIMAP_HEIGHT = 0  
    RATIO_TO_DETERMINED_SCOUTED = 0.5

    SIZE = ENEMY_BUILDING_MAT_END

class ACTIONS:
    
    ACTION_DO_NOTHING = 0
    ACTION_DEVELOP_BASE = 1
    ACTION_ATTACK = 2
    ACTION_SCOUT_START = 3
    ACTION_SCOUT_END = ACTION_SCOUT_START + STATE.GRID_SIZE * STATE.GRID_SIZE
    ACTION_ATTACK_START = ACTION_SCOUT_END
    ACTION_ATTACK_END = ACTION_ATTACK_START + STATE.GRID_SIZE * STATE.GRID_SIZE
    SIZE = ACTION_ATTACK_END

    ACTIONS2SUB_AGENTSID = {}
    ACTIONS2SUB_AGENTSID[ACTION_DO_NOTHING] = SUB_AGENT_ID_DONOTHING
    ACTIONS2SUB_AGENTSID[ACTION_DEVELOP_BASE] = SUB_AGENT_ID_BASE
    
    for a in range(ACTION_SCOUT_START, ACTION_SCOUT_END):
        ACTIONS2SUB_AGENTSID[a] = SUB_AGENT_ID_SCOUT

    for a in range(ACTION_ATTACK_START, ACTION_ATTACK_END):
        ACTIONS2SUB_AGENTSID[a] = SUB_AGENT_ID_ATTACK

    ACTION2STR = {}
    ACTION2STR[ACTION_DO_NOTHING] = "DoNothing"
    ACTION2STR[ACTION_DEVELOP_BASE] = "Develop_Base"
    ACTION2STR[ACTION_ATTACK] = "Attack"
    for a in range(ACTION_SCOUT_START, ACTION_SCOUT_END):
        ACTION2STR[a] = "Scout_" + str(a - ACTION_SCOUT_START)

    for a in range(ACTION_ATTACK_START, ACTION_ATTACK_END):
        ACTION2STR[a] = "Attack_" + str(a - ACTION_ATTACK_START)


# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE.SIZE, ACTIONS.SIZE)
RUN_TYPES[QTABLE][DIRECTORY] = "superAgent_qtable"
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "qtable"
RUN_TYPES[QTABLE][HISTORY] = "replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE.SIZE, ACTIONS.SIZE)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "superAgent_dqn"
RUN_TYPES[DQN][DIRECTORY] = "superAgent_dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "result"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"

class SharedDataSuper(SharedDataBase, SharedDataBattle):
    def __init__(self):
        super(SharedDataSuper, self).__init__()
    
class SuperAgent(BaseAgent):
    def __init__(self, runArg = None, decisionMaker = None, isMultiThreaded = False, playList = None, trainList = None):
        super(SuperAgent, self).__init__()

        self.trainAgent = AGENT_NAME in trainList
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList
            

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(runArg, isMultiThreaded)

        # create sub agents and get decision makers
        self.subAgents = {}

        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)
            saArg = SUBAGENTS_ARGS[key]
            if saArg == "inherit":
                saArg = runArg

            self.subAgents[key] = saClass(saArg, saDM, isMultiThreaded, saPlayList, trainList)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()
            self.activeSubAgents = [self.subAgentPlay]
            if self.subAgentPlay == SUB_AGENT_ID_BASE:
                self.activeSubAgents.append(SUB_AGENT_ID_DONOTHING)
        else:
            self.activeSubAgents = list(range(NUM_SUB_AGENTS))


        # model params 
        self.terminalState = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

    def CreateDecisionMaker(self, runArg, isMultiThreaded):

        if runArg == None:
            runTypeArg = list(ALL_TYPES.intersection(sys.argv))
            runArg = runTypeArg.pop()    
        runType = RUN_TYPES[runArg]

        decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=AGENT_DIR + runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1

        for key, sa in self.subAgents.items():
            if sa.FindActingHeirarchi() >= 0:
                return key
        
        return -1

    def step(self, obs):
        super(SuperAgent, self).step(obs)
        
        try:
            self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]
            
            if obs.last():
                self.LastStep(obs)
                return SC2_Actions.DO_NOTHING_SC2_ACTION
            elif obs.first():
                self.FirstStep(obs)
            
            time.sleep(STEP_DURATION)
            for sa in self.activeSubAgents:
                self.subAgentsActions[sa] = self.subAgents[sa].step(obs, self.sharedData, self.move_number)


            if self.move_number == 0:
                self.CreateState(obs)
                self.Learn()

                self.current_action = self.ChooseAction()
                #print("valid actions =", self.ValidActions())
                #print("\nactionChosen =", self.Action2Str(False), "\nactionActed =", self.Action2Str(True))
                # self.PrintState()
            
            self.step_num += 1
            sc2Action = self.ActAction(obs)
            return sc2Action

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def FirstStep(self, obs):
        self.move_number = 0
        self.step_num = 0 
        self.sharedData = SharedDataSuper()

        player_y, player_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()

        if player_y.any() and player_y.mean() <= 31:
            self.base_top_left = True 
        else:
            self.base_top_left = False

        # actions:
        self.current_action = None

        self.subAgentsActions = {}
        for sa in range(NUM_SUB_AGENTS):
            self.subAgentsActions[sa] = None

        # states:
        self.previous_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        self.sharedData.currBaseState = np.zeros(STATE.BASE_END - STATE.BASE_START, dtype=np.int32, order='C')



    def LastStep(self, obs):
        reward = obs.reward
        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(self.current_state.copy(), self.current_action, reward, self.terminalState.copy(), True)

            score = obs.observation["score_cumulative"][0]
            self.decisionMaker.end_run(reward, score, self.step_num)
        
        for sa in self.activeSubAgents:
            self.subAgents[sa].LastStep(obs, reward) 


    def ChooseAction(self):
        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions()
                if self.trainAgent:
                    targetValues = False
                    exploreProb = self.decisionMaker.ExploreProb()              
                else:
                    targetValues = True
                    exploreProb = 0   

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

        return action

    def ValidActions(self):
        
        valid = [ACTIONS.ACTION_DO_NOTHING, ACTIONS.ACTION_DEVELOP_BASE]
        
        hasArmy = False
        for key, val in self.sharedData.armySize.items():
            if val > 0:
                hasArmy = True
                break
        
        if hasArmy:
            for i in range(STATE.GRID_SIZE * STATE.GRID_SIZE):
                if self.current_scaled_state[STATE.FOG_MAT_START + i] > 0:
                    valid.append(ACTIONS.ACTION_SCOUT_START + i)
                if self.current_scaled_state[STATE.ENEMY_ARMY_MAT_START + i] > 0:
                    valid.append(ACTIONS.ACTION_ATTACK_START + i)

        return valid

    def Action2Str(self, realAct):
        if realAct:
            subAgent, subAgentAction = self.GetAction2Act()
        else:
            subAgent, subAgentAction = self.AdjustAction2SubAgents()

        return ACTIONS.ACTION2STR[subAgent] + "-->" + self.subAgents[subAgent].Action2Str(subAgentAction)

    def ActAction(self, obs): 
        subAgent, subAgentAction = self.GetAction2Act()
        sc2Action, terminal = self.subAgents[subAgent].Action2SC2Action(obs, subAgentAction, self.move_number)
        
        if terminal:
            self.move_number = 0
        else:
            self.move_number += 1

        return sc2Action

    def CreateState(self, obs):
        for si in range(STATE.BASE_START, STATE.BASE_END):
            self.current_state[si] = self.sharedData.currBaseState[si]

        miniMapVisi = obs.observation['minimap'][SC2_Params.VISIBILITY_MINIMAP]
        miniMapHeight = obs.observation['minimap'][SC2_Params.HEIGHT_MAP]

        visibilityCount = []
        for i in range(STATE.GRID_SIZE * STATE.GRID_SIZE):
            visibilityCount.append([0,0,0])

        for y in range(SC2_Params.MINIMAP_SIZE):
            pixY = int(y / (SC2_Params.MINIMAP_SIZE / STATE.GRID_SIZE))
            for x in range(SC2_Params.MINIMAP_SIZE):
                if miniMapHeight[y][x] != STATE.NON_VALID_MINIMAP_HEIGHT:
                    pixX = int(x / (SC2_Params.MINIMAP_SIZE / STATE.GRID_SIZE))
                    idx = pixX + pixY * STATE.GRID_SIZE
                    visibilityCount[idx][miniMapVisi[y][x]] += 1
        
        for y in range(STATE.GRID_SIZE):
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                allCount = sum(visibilityCount[idx])
                
                if allCount == 0:
                    allCount += 0.1
                
                ratio = visibilityCount[idx][2] / allCount

                if ratio > STATE.RATIO_TO_DETERMINED_SCOUTED:
                    self.current_state[idx + STATE.FOG_MAT_START] = 0
                else:
                    self.current_state[idx + STATE.FOG_MAT_START] = 1



        self.ScaleCurrState()

        

    def AdjustAction2SubAgents(self):
        subAgentIdx = self.current_action

        if self.current_action >= ACTIONS.ACTION_SCOUT_START and self.current_action < ACTIONS.ACTION_ATTACK_END:
            subAgentIdx = SUB_AGENT_ID_SCOUT
            self.subAgentsActions[subAgentIdx] = self.current_action - ACTIONS.ACTION_SCOUT_START
        elif self.current_action == ACTIONS.ACTION_ATTACK:
            subAgentIdx = SUB_AGENT_ID_ATTACK

        return subAgentIdx, self.subAgentsActions[subAgentIdx]

    def GetAction2Act(self):
        subAgentIdx, subAgentAction = self.AdjustAction2SubAgents()
        
        if SUB_AGENT_ID_DONOTHING in self.activeSubAgents and self.subAgents[subAgentIdx].IsDoNothingAction(subAgentAction):
            subAgentIdx = ACTIONS.ACTION_DO_NOTHING
            subAgentAction = self.subAgentsActions[subAgentIdx]

        return subAgentIdx, subAgentAction

    def Learn(self, reward = 0):
        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(str(self.previous_scaled_state), self.current_action, reward, str(self.current_scaled_state))

        for sa in self.activeSubAgents:
            self.subAgents[sa].Learn(reward)
     
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

    def PrintState(self):
        self.subAgents[SUB_AGENT_ID_BASE].PrintState()
        print("self mat\t fog mat\t enemy mat")
        for y in range(STATE.GRID_SIZE):
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.SELF_MAT_START + idx], end = ' ')
            
            print(end = "   |   ")
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.FOG_MAT_START + idx], end = ' ')

            print(end = "   |   ")
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.ENEMY_ARMY_MAT_START + idx], end = ' ')

            print("||")
  
