import random
import math
import os.path
import datetime
import time
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


from pysc2.lib import actions
from pysc2.lib.units import Terran

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

from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import GetScreenCorners

AGENT_DIR = "SuperAgent/"

AGENT_NAME = "super"

basic_DM_Types = {'super': 'dqn', 'base_mngr': 'dqn_Embedding', 'battle_mngr': 'naive', 'do_nothing': 'naive', 'army_attack': 'dqn_Embedding', 'base_attack': 'naive', 'builder': 'dqn_Embedding', 'trainer': 'dqn_Embedding'}


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

    
    BASE_START = 0
    BASE_END = BASE.SIZE

    # power and fog mat
    SELF_POWER_START = BASE_END
    SELF_POWER_END = SELF_POWER_START + GRID_SIZE * GRID_SIZE
    FOG_MAT_START = SELF_POWER_END
    FOG_MAT_END = FOG_MAT_START + GRID_SIZE * GRID_SIZE
    FOG_COUNTER_MAT_START = FOG_MAT_END
    FOG_COUNTER_MAT_END = FOG_COUNTER_MAT_START + GRID_SIZE * GRID_SIZE

    ENEMY_ARMY_MAT_START = FOG_MAT_END
    ENEMY_ARMY_MAT_END = ENEMY_ARMY_MAT_START + GRID_SIZE * GRID_SIZE
    
    ENEMY_BUILDING_MAT_START = ENEMY_ARMY_MAT_END
    ENEMY_BUILDING_MAT_END = ENEMY_BUILDING_MAT_START + GRID_SIZE * GRID_SIZE  

    TIME_LINE_IDX = ENEMY_BUILDING_MAT_END

    SIZE = TIME_LINE_IDX + 1

    MAX_SCOUT_VAL = 10
    VAL_IS_SCOUTED = 8

    TIME_LINE_BUCKETING = 20


class ACTIONS:
    
    ACTION_DO_NOTHING = 0
    ACTION_DEVELOP_BASE = 1
    ACTION_SCOUT_START = 2
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
        self.numStep = 0
        self.numAgentStep = 0


NORMALIZATION_TRAIN_LOCAL_REWARD = 20
NORMALIZATION_TRAIN_GAME_REWARD = 100

    
class SuperAgent(BaseAgent):
    def __init__(self, dmTypes, useMapRewards, decisionMaker = None, isMultiThreaded = False, playList = None, trainList = None):
        super(SuperAgent, self).__init__(STATE.SIZE)

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
        self.unitPower[Terran.Marine] = sum(valVecMarine) / len(valVecMarine)
        self.unitPower[Terran.Reaper] = sum(table.ix['reaper', :]) / len(valVecMarine)
        self.unitPower[Terran.Hellion] = sum(table.ix['hellion', :]) / len(valVecMarine)
        self.unitPower[Terran.SiegeTank] = sum(table.ix['siege tank', :]) / len(valVecMarine)

        self.useMapRewards = useMapRewards

        self.move_number = 0

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

        self.unit_type = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        if obs.first():
            self.FirstStep(obs)
            return self.SendAction(self.go2BaseAction)
        elif self.sharedData.numStep == 1:
            # get default screen location
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            self.baseNorthWestScreenCorner = cameraCornerNorthWest
        elif obs.last():
            self.LastStep(obs)
            return SC2_Actions.DO_NOTHING_SC2_ACTION

        time.sleep(STEP_DURATION)

        self.MonitorObservation(obs)
        if self.move_number == 0:
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            
            if cameraCornerNorthWest != self.baseNorthWestScreenCorner:
                print("\nnot in base location:", cameraCornerNorthWest, "!=", self.baseNorthWestScreenCorner)
                return self.SendAction(self.go2BaseAction)

            self.sharedData.numAgentStep += 1
            self.CreateState(obs)
            self.Learn()
            self.ChooseAction()
            # if self.playAgent:
            #     print("\nactionChosen =", self.Action2Str(False), "\nactionActed =", self.Action2Str(True))
            #     self.PrintState()
        
        sc2Action = self.ActAction(obs)

        return self.SendAction(sc2Action)

    def FirstStep(self, obs):
        self.move_number = 0
        self.sharedData.numStep = 0 
        self.sharedData.numAgentStep = 0

        self.sharedData.__init__()
        
        cc_y, cc_x = (self.unit_type == Terran.CommandCenter).nonzero()
        if len(cc_y) > 0:
            middleCC = FindMiddle(cc_y, cc_x)
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            miniMapLoc = Scale2MiniMap(middleCC, cameraCornerNorthWest, cameraCornerSouthEast)
            self.sharedData.commandCenterLoc = [miniMapLoc]
            self.sharedData.buildingCount[Terran.CommandCenter] += 1   
            print("first step North East =", cameraCornerNorthWest)


        self.sharedData.unitTrainValue = self.unitPower
        self.sharedData.currBaseState = np.zeros(STATE.BASE_END - STATE.BASE_START, dtype=np.int32, order='C')

        # actions:
        self.current_action = None

        # states:
        self.current_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE.SIZE, dtype=np.int32, order='C')

        if SUB_AGENT_ID_BASE in self.activeSubAgents and not self.trainAgent:
            self.accumulatedTrainReward = 0.0
            self.discountForLocalReward = 0.99

        self.subAgentsActions = {}
        for sa in range(NUM_SUB_AGENTS):
            self.subAgentsActions[sa] = None
            self.subAgents[sa].FirstStep(obs)
        
        coord = self.sharedData.commandCenterLoc[0]

        self.go2BaseAction = actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(coord)])

    def LastStep(self, obs):
        if self.useMapRewards:
            reward = obs.reward
        else:
            reward = 0

        baseReward =reward
        if SUB_AGENT_ID_BASE in self.activeSubAgents and not self.trainAgent:
            reward += self.accumulatedTrainReward / NORMALIZATION_TRAIN_GAME_REWARD
        
        print("\nbase reward =", baseReward, "real reward =", reward, "accumulatedReward =", self.accumulatedTrainReward)

        score = obs.observation["score_cumulative"][0]

        self.CreateState(obs)
        self.Learn(reward, True)  
        self.EndRun(reward, score, self.sharedData.numStep)

    def SendAction(self, sc2Action):
        self.sharedData.numStep += 1
        return sc2Action

    def EndRun(self, reward, score, numStep):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, numStep)
        
        for sa in self.activeSubAgents:
            self.subAgents[sa].EndRun(reward, score, numStep) 

    def ChooseAction(self):
        for sa in self.activeSubAgents:
                self.subAgentsActions[sa] = self.subAgents[sa].ChooseAction()

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
                action = ACTIONS.ACTION_ATTACK_START
            else:
                action = self.subAgentPlay

        self.current_action = action
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
        subAgent, subAgentAction = self.AdjustAction2SubAgents()
        if realAct:
            subAgent, subAgentAction = self.GetAction2Act(subAgent, subAgentAction)

        return SUBAGENTS_NAMES[subAgent] + "-->" + self.subAgents[subAgent].Action2Str(subAgentAction)

    def ActAction(self, obs): 
        # get subagent and action
        subAgent, subAgentAction = self.AdjustAction2SubAgents()
        # mark to sub agent that his action was chosen
        self.subAgents[subAgent].SubAgentActionChosen(subAgentAction)
        # find real action to act
        subAgent, subAgentAction = self.GetAction2Act(subAgent, subAgentAction)
        # preform action
        sc2Action, terminal = self.subAgents[subAgent].Action2SC2Action(obs, subAgentAction, self.move_number)

        if terminal:
            self.move_number = 0
        else:
            self.move_number += 1

        return sc2Action


    def MonitorObservation(self, obs):
        for sa in self.activeSubAgents:
            self.subAgents[sa].MonitorObservation(obs)
                     
    def CreateState(self, obs):
        for subAgent in self.subAgents.values():
            subAgent.CreateState(obs)

        for si in range(STATE.BASE_START, STATE.BASE_END):
            self.current_state[si] = self.sharedData.currBaseState[si]

        for y in range(STATE.GRID_SIZE):
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
               
                self.current_state[idx + STATE.ENEMY_ARMY_MAT_START] = self.sharedData.enemyMatObservation[y, x]
                self.current_state[idx + STATE.FOG_MAT_START] = int( self.sharedData.fogRatioMat[y, x] * STATE.MAX_SCOUT_VAL)
                self.current_state[idx + STATE.FOG_COUNTER_MAT_START] = self.sharedData.fogCounterMat[y, x]

        self.current_state[STATE.TIME_LINE_IDX] = self.sharedData.numStep    
        self.ScaleCurrState()

    def AdjustAction2SubAgents(self):
        subAgentIdx = ACTIONS.ACTIONS2SUB_AGENTSID[self.current_action]
    
        if subAgentIdx == SUB_AGENT_ID_SCOUT:
            self.subAgentsActions[subAgentIdx] = self.current_action - ACTIONS.ACTION_SCOUT_START
        elif subAgentIdx == SUB_AGENT_ID_ATTACK:            
            self.subAgentsActions[subAgentIdx] = self.current_action - ACTIONS.ACTION_ATTACK_START

        return subAgentIdx, self.subAgentsActions[subAgentIdx]

    def GetAction2Act(self, subAgentIdx, subAgentAction):  
        if SUB_AGENT_ID_DONOTHING in self.activeSubAgents and self.subAgents[subAgentIdx].IsDoNothingAction(subAgentAction):
            subAgentIdx = ACTIONS.ACTION_DO_NOTHING
            subAgentAction = self.subAgentsActions[subAgentIdx]

        return subAgentIdx, subAgentAction

    def Learn(self, reward = 0, terminal = False):
        
        if SUB_AGENT_ID_BASE in self.activeSubAgents and not self.trainAgent:
            localReward = self.sharedData.prevTrainActionReward * pow(self.discountForLocalReward, self.sharedData.numAgentStep)
            reward = localReward / NORMALIZATION_TRAIN_LOCAL_REWARD + reward
            self.accumulatedTrainReward += localReward
            self.sharedData.prevTrainActionReward = 0

        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(self.previous_scaled_state, self.current_action, reward, self.current_scaled_state, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]

        for sa in self.activeSubAgents:
            self.subAgents[sa].Learn(reward, terminal)
     
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

        self.current_scaled_state[STATE.TIME_LINE_IDX] = int(self.current_state[STATE.TIME_LINE_IDX] / STATE.TIME_LINE_BUCKETING)

    def PrintState(self):
        self.subAgents[SUB_AGENT_ID_BASE].PrintState()

        print("self mat     fog mat     enemy mat")
        for y in range(STATE.GRID_SIZE):
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.SELF_POWER_START + idx], end = ' ')

            print(end = "   |   ")
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.FOG_MAT_START + idx], end = ',')
                print(self.current_scaled_state[STATE.FOG_COUNTER_MAT_START + idx], end = ' ')

            print(end = "   |   ")
            for x in range(STATE.GRID_SIZE):
                idx = x + y * STATE.GRID_SIZE
                print(self.current_scaled_state[STATE.ENEMY_ARMY_MAT_START + idx], end = ' ')

            print("||")
  

if __name__ == "__main__":
    if "results" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES)