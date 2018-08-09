import random
import math
import os.path
import logging
import traceback
import datetime
import time
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


from pysc2.lib import actions

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_decisionMaker import LearnWithReplayMngr

from utils_qtable import QTableParamsExplorationDecay
from utils_dqn import DQN_PARAMS

from utils_results import PlotResults

# sub agents

from agent_do_nothing import DoNothingSubAgent
from agent_base_mngr import BaseMngr
from agent_scout import ScoutAgent
from agent_attack import AttackAgent

# shared data
from agent_base_mngr import SharedDataBase
from agent_attack import SharedDataAttack
from agent_scout import SharedDataScout

from agent_base_mngr import BASE_STATE

from utils import FindMiddle
from utils import Scale2MiniMap
from utils import GetScreenCorners

AGENT_DIR = "SuperAgent/"

AGENT_NAME = "super"

basic_DM_Types = {'super': 'dqn', 'base_mngr': 'dqn_Embedding', 'battle_mngr': 'naive', 'do_nothing': 'naive', 'army_attack': 'dqn_Embedding', 'base_attack': 'naive', 'builder': 'dqn_Embedding', 'trainer': 'dqn_Embedding'}


STEP_DURATION = 0.0

SUB_AGENT_ID_DONOTHING = 0
SUB_AGENT_ID_BASE = 1
SUB_AGENT_ID_ATTACK = 2
SUB_AGENT_ID_SCOUT = 3
NUM_SUB_AGENTS = 4

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_ID_DONOTHING] = "DoNothingSubAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_BASE] = "BaseMngr"
SUBAGENTS_NAMES[SUB_AGENT_ID_SCOUT] = "ScoutAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_ATTACK] = "AttackAgent"

# possible types of play

QTABLE = 'q'
DQN = 'dqn'
DQN_2L = "dqn_2l"

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
    SELF_ATTACK_POWER = BASE_END
    FOG_MAT_START = SELF_ATTACK_POWER + 1
    FOG_MAT_END = FOG_MAT_START + GRID_SIZE * GRID_SIZE
    ENEMY_ARMY_MAT_START = FOG_MAT_END
    ENEMY_ARMY_MAT_END = ENEMY_ARMY_MAT_START + GRID_SIZE * GRID_SIZE
    ENEMY_BUILDING_MAT_START = ENEMY_ARMY_MAT_END
    ENEMY_BUILDING_MAT_END = ENEMY_BUILDING_MAT_START + GRID_SIZE * GRID_SIZE  

    NON_VALID_MINIMAP_HEIGHT = 0  
    MAX_SCOUT_VAL = 10
    VAL_IS_SCOUTED = 8

    SIZE = ENEMY_BUILDING_MAT_END

class ACTIONS:
    
    ACTION_DO_NOTHING = 0
    ACTION_DEVELOP_BASE = 1
    ACTION_SCOUT_START = 2
    ACTION_SCOUT_END = ACTION_SCOUT_START + STATE.GRID_SIZE * STATE.GRID_SIZE
    ACTION_ATTACK_PREFORM = ACTION_SCOUT_END
    ACTION_ATTACK_START = ACTION_ATTACK_PREFORM + 1
    ACTION_ATTACK_END = ACTION_ATTACK_START + STATE.GRID_SIZE * STATE.GRID_SIZE
    SIZE = ACTION_ATTACK_END

    ACTIONS2SUB_AGENTSID = {}
    ACTIONS2SUB_AGENTSID[ACTION_DO_NOTHING] = SUB_AGENT_ID_DONOTHING
    ACTIONS2SUB_AGENTSID[ACTION_DEVELOP_BASE] = SUB_AGENT_ID_BASE
    
    for a in range(ACTION_SCOUT_START, ACTION_SCOUT_END):
        ACTIONS2SUB_AGENTSID[a] = SUB_AGENT_ID_SCOUT

    ACTIONS2SUB_AGENTSID[ACTION_ATTACK_PREFORM] = SUB_AGENT_ID_ATTACK
    for a in range(ACTION_ATTACK_START, ACTION_ATTACK_END):
        ACTIONS2SUB_AGENTSID[a] = SUB_AGENT_ID_ATTACK

    ACTION2STR = {}
    ACTION2STR[ACTION_DO_NOTHING] = "DoNothing"
    ACTION2STR[ACTION_DEVELOP_BASE] = "Develop_Base"
    ACTION2STR[ACTION_ATTACK_PREFORM] = "PreformAttack"
    for a in range(ACTION_SCOUT_START, ACTION_SCOUT_END):
        ACTION2STR[a] = "Scout_" + str(a - ACTION_SCOUT_START)

    for a in range(ACTION_ATTACK_START, ACTION_ATTACK_END):
        ACTION2STR[a] = "Attack_" + str(a - ACTION_ATTACK_START)


# Define the neural network
def dqn_2layers(inputLayer, numActions, scope):
    with tf.variable_scope(scope):
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 512)
        fc2 = tf.contrib.layers.fully_connected(fc1, 512)
        output = tf.contrib.layers.fully_connected(fc2, numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
        
    return output
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

RUN_TYPES[DQN_2L] = {}
RUN_TYPES[DQN_2L][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN_2L][PARAMS] = DQN_PARAMS(STATE.SIZE, ACTIONS.SIZE, nn_Func=dqn_2layers)
RUN_TYPES[DQN_2L][DIRECTORY] = "superAgent_dqn2l"
RUN_TYPES[DQN_2L][DECISION_MAKER_NAME] = "superAgent_dqn2l"
RUN_TYPES[DQN_2L][HISTORY] = "replayHistory"
RUN_TYPES[DQN_2L][RESULTS] = "result"



RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"

UNIT_VALUE_TABLE_NAME = 'unit_value_table.gz'

class SharedDataSuper(SharedDataBase, SharedDataAttack, SharedDataScout):
    def __init__(self):
        super(SharedDataSuper, self).__init__()
    
class SuperAgent(BaseAgent):
    def __init__(self, dmTypes, decisionMaker = None, isMultiThreaded = False, playList = None, trainList = None):
        super(SuperAgent, self).__init__()

        self.sharedData = SharedDataSuper()

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
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        # create sub agents and get decision makers
        self.subAgents = {}

        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)
            self.subAgents[key] = saClass(self.sharedData, dmTypes, saDM, isMultiThreaded, saPlayList, trainList)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()
            self.activeSubAgents = [self.subAgentPlay]
            if self.subAgentPlay == SUB_AGENT_ID_BASE:
                self.activeSubAgents.append(SUB_AGENT_ID_DONOTHING)
        else:
            self.activeSubAgents = list(range(NUM_SUB_AGENTS))

        self.unitPower = {}
        table = pd.read_pickle(UNIT_VALUE_TABLE_NAME, compression='gzip')
        valVecMarine = table.ix['marine', :]
        self.unitPower[TerranUnit.MARINE] = sum(valVecMarine) / len(valVecMarine)
        self.unitPower[TerranUnit.REAPER] = sum(table.ix['reaper', :]) / len(valVecMarine)
        self.unitPower[TerranUnit.HELLION] = sum(table.ix['hellion', :]) / len(valVecMarine)
        self.unitPower[TerranUnit.SIEGE_TANK] = sum(table.ix['siege tank', :]) / len(valVecMarine)

        # model params 
        self.terminalState = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        
        runType = RUN_TYPES[dmTypes[AGENT_NAME]]
        # create proj directory
        if not os.path.isdir("./" + dmTypes["directory"]):
            os.makedirs("./" + dmTypes["directory"])

        # create agent dir
        directory = dmTypes["directory"] + "/" + AGENT_DIR
        if not os.path.isdir("./" + directory):
            os.makedirs("./" + directory)
        decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], numTrials2Learn=50, 
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory + runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

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
                self.subAgentsActions[sa] = self.subAgents[sa].step(obs, self.move_number)


            if self.move_number == 0:
                self.CreateState(obs)
                self.Learn()

                self.current_action = self.ChooseAction()
                # if self.playAgent:
                #     print("valid actions =", self.ValidActions())
                #     print("\nactionChosen =", self.Action2Str(False), "\nactionActed =", self.Action2Str(True))
                #     self.PrintState()
            
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

        self.sharedData.__init__()
        
        cc_y, cc_x = (self.unit_type == TerranUnit.COMMANDCENTER).nonzero()
        if len(cc_y) > 0:
            middleCC = FindMiddle(cc_y, cc_x)
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            miniMapLoc = Scale2MiniMap(middleCC, cameraCornerNorthWest , cameraCornerSouthEast)
            self.sharedData.CommandCenterLoc = [miniMapLoc]
    
        self.sharedData.unitTrainValue = self.unitPower
        self.sharedData.buildingCount[TerranUnit.COMMANDCENTER] += 1        
        self.sharedData.currBaseState = np.zeros(STATE.BASE_END - STATE.BASE_START, dtype=np.int32, order='C')

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


        self.scoutingInfo = [0.0, 0.0, 0.0, 0.0]

    def LastStep(self, obs):
        reward = obs.reward
        if self.trainAgent:
            self.CreateState(obs)
            self.Learn(reward, True)

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
            if self.subAgentPlay == SUB_AGENT_ID_ATTACK:
                action = ACTIONS.ACTION_ATTACK_PREFORM
            else:
                action = self.subAgentPlay

        return action

    def ValidActions(self):
        
        valid = [ACTIONS.ACTION_DO_NOTHING, ACTIONS.ACTION_DEVELOP_BASE]

        if self.current_scaled_state[STATE.BASE.ARMY_POWER] > 0:
            for i in range(STATE.GRID_SIZE * STATE.GRID_SIZE):
                if self.current_scaled_state[STATE.FOG_MAT_START + i] < STATE.VAL_IS_SCOUTED:
                    valid.append(ACTIONS.ACTION_SCOUT_START + i)
                if self.current_scaled_state[STATE.ENEMY_ARMY_MAT_START + i] > 0:
                    valid.append(ACTIONS.ACTION_ATTACK_START + i)

        return valid

    def Action2Str(self, realAct):
        if realAct:
            subAgent, subAgentAction = self.GetAction2Act()
        else:
            subAgent, subAgentAction = self.AdjustAction2SubAgents()


        return SUBAGENTS_NAMES[subAgent] + "-->" + self.subAgents[subAgent].Action2Str(subAgentAction)

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


        power = 0.0
        for unit, num in self.sharedData.armyInAttack.items():
            if unit in self.sharedData.unitTrainValue:
                power += num * self.sharedData.unitTrainValue[unit]

        self.current_state[STATE.SELF_ATTACK_POWER] = power

        enemyMat = obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] == SC2_Params.PLAYER_HOSTILE
        miniMapVisi = obs.observation['minimap'][SC2_Params.VISIBILITY_MINIMAP]
        miniMapHeight = obs.observation['minimap'][SC2_Params.HEIGHT_MAP]

        visibilityCount = []
        for i in range(STATE.GRID_SIZE * STATE.GRID_SIZE):
            visibilityCount.append([0,0,0])
            self.current_state[i + STATE.ENEMY_ARMY_MAT_START] = 0

        for y in range(SC2_Params.MINIMAP_SIZE):
            pixY = int(y / (SC2_Params.MINIMAP_SIZE / STATE.GRID_SIZE))
            for x in range(SC2_Params.MINIMAP_SIZE):
                pixX = int(x / (SC2_Params.MINIMAP_SIZE / STATE.GRID_SIZE))
                idx = pixX + pixY * STATE.GRID_SIZE
                self.current_state[idx + STATE.ENEMY_ARMY_MAT_START] += enemyMat[y][x]
                
                if miniMapHeight[y][x] != STATE.NON_VALID_MINIMAP_HEIGHT:
                    visibilityCount[idx][miniMapVisi[y][x]] += 1
                

        
        for y in range(STATE.GRID_SIZE):
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE  
                allCount = sum(visibilityCount[idx])
                allCount = 0.1 if allCount == 0 else allCount
                
                ratio = (visibilityCount[idx][2] + visibilityCount[idx][1]) / allCount
                self.current_state[idx + STATE.FOG_MAT_START] = round(ratio * STATE.MAX_SCOUT_VAL)

        self.ScaleCurrState()

        

    def AdjustAction2SubAgents(self):
        subAgentIdx = ACTIONS.ACTIONS2SUB_AGENTSID[self.current_action]
    
        if subAgentIdx == SUB_AGENT_ID_SCOUT:
            self.subAgentsActions[subAgentIdx] = self.current_action - ACTIONS.ACTION_SCOUT_START
        elif subAgentIdx == SUB_AGENT_ID_ATTACK:            
            self.subAgentsActions[subAgentIdx] = self.current_action - ACTIONS.ACTION_ATTACK_PREFORM

        return subAgentIdx, self.subAgentsActions[subAgentIdx]

    def GetAction2Act(self):
        subAgentIdx, subAgentAction = self.AdjustAction2SubAgents()
        
        if SUB_AGENT_ID_DONOTHING in self.activeSubAgents and self.subAgents[subAgentIdx].IsDoNothingAction(subAgentAction):
            subAgentIdx = ACTIONS.ACTION_DO_NOTHING
            subAgentAction = self.subAgentsActions[subAgentIdx]

        return subAgentIdx, subAgentAction

    def Learn(self, reward = 0, terminal = False):
        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(self.previous_scaled_state, self.current_action, reward, self.current_scaled_state, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        for sa in self.activeSubAgents:
            self.subAgents[sa].Learn(reward)
     
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

    def PrintState(self):
        self.subAgents[SUB_AGENT_ID_BASE].PrintState()
        print("attack power =", self.current_scaled_state[STATE.SELF_ATTACK_POWER])
        print("fog mat\tenemy mat")
        for y in range(STATE.GRID_SIZE):
            print(end = "   |   ")
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.FOG_MAT_START + idx], end = ' ')

            print(end = "   |   ")
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.ENEMY_ARMY_MAT_START + idx], end = ' ')

            print("||")
  

if __name__ == "__main__":
    if "results" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES)