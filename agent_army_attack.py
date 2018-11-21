# build base sub agent
import sys
import random
import math
import time
import os.path
import datetime
import threading

import numpy as np
import pandas as pd

from pysc2.lib import actions

from utils import BaseAgent
from utils import EmptySharedData

import tensorflow as tf

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from paramsCalibration import ChangeParamsAccordingToDict

#decision makers
from algo_decisionMaker import DecisionMakerExperienceReplay
from algo_decisionMaker import DecisionMakerOnlineAsync
from algo_decisionMaker import UserPlay
from algo_decisionMaker import BaseNaiveDecisionMaker
from algo_decisionMaker import BaseDecisionMaker

# params
from algo_dqn import DQN_PARAMS
from algo_dqn import DQN_EMBEDDING_PARAMS
from algo_a2c import A2C_PARAMS
from algo_a3c import A3C_PARAMS
from algo_qtable import QTableParams
from algo_qtable import QTableParamsExplorationDecay

from utils_results import PlotResults

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints

AGENT_DIR = "ArmyAttack/"
AGENT_NAME = "army_attack"

NUM_TRIALS_2_SAVE = 100

GRID_SIZE = 5

ACTION_DO_NOTHING = 0
ACTIONS_START_IDX_ATTACK = 1
ACTIONS_END_IDX_ATTACK = ACTIONS_START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
NUM_ACTIONS = ACTIONS_END_IDX_ATTACK

ACTION2STR = {}

ACTION2STR[ACTION_DO_NOTHING] = "Do_Nothing"
for a in range(ACTIONS_START_IDX_ATTACK, ACTIONS_END_IDX_ATTACK):
    ACTION2STR[a] = "ArmyAttack_" + str(a - ACTIONS_START_IDX_ATTACK)

STATE_START_SELF_MAT = 0
STATE_END_SELF_MAT = STATE_START_SELF_MAT + GRID_SIZE * GRID_SIZE
STATE_START_ENEMY_MAT = STATE_END_SELF_MAT
STATE_END_ENEMY_MAT = STATE_START_ENEMY_MAT + GRID_SIZE * GRID_SIZE
STATE_TIME_LINE_IDX = STATE_END_ENEMY_MAT
STATE_SIZE = STATE_TIME_LINE_IDX + 1


# possible types of decision maker
QTABLE = 'qtable'
DQN = 'dqn'
DQN2L = 'dqn_2l'
DQN2L_EXPLORATION_CHANGE = 'dqn_2l_explorationChange'
A2C = 'A2C'
A3C = 'A3C'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding'
NAIVE = 'naive' 

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, DQN_EMBEDDING_LOCATIONS])

# data for run type
TYPE = "type"
DECISION_MAKER_TYPE = "dm_type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "history"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

# table names

RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][DIRECTORY] = "armyAttack_q"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE_SIZE, NUM_ACTIONS, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "armyAttack_q_qtable"
RUN_TYPES[QTABLE][HISTORY] = "armyAttack_q_replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "armyAttack_q_result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][DIRECTORY] = "armyAttack_dqn"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "armyAttack_dqn"
RUN_TYPES[DQN][HISTORY] = "armyAttack_dqn_replayHistory"
RUN_TYPES[DQN][RESULTS] = "armyAttack_dqn_result"


RUN_TYPES[DQN2L] = {}
RUN_TYPES[DQN2L][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[DQN2L][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN2L][DIRECTORY] = "armyAttack_dqn2l"
RUN_TYPES[DQN2L][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, layersNum=2, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[DQN2L][DECISION_MAKER_NAME] = "armyAttack_dqn2l"
RUN_TYPES[DQN2L][HISTORY] = "armyAttack_dqn2l_replayHistory"
RUN_TYPES[DQN2L][RESULTS] = "armyAttack_dqn2l_result"


RUN_TYPES[DQN2L_EXPLORATION_CHANGE] = {}
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][DIRECTORY] = "armyAttack_dqn2l"
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, layersNum=2, explorationProb=0.01, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][DECISION_MAKER_NAME] = "armyAttack_dqn2l"
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][HISTORY] = "armyAttack_dqn2l_replayHistory"
RUN_TYPES[DQN2L_EXPLORATION_CHANGE][RESULTS] = "armyAttack_dqn2l_result"

RUN_TYPES[A2C] = {}
RUN_TYPES[A2C][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[A2C][TYPE] = "A2C"
RUN_TYPES[A2C][DIRECTORY] = "armyAttack_A2C"
RUN_TYPES[A2C][PARAMS] = A2C_PARAMS(STATE_SIZE, NUM_ACTIONS, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[A2C][DECISION_MAKER_NAME] = "armyAttack_A2C"
RUN_TYPES[A2C][HISTORY] = "armyAttack_A2C_replayHistory"
RUN_TYPES[A2C][RESULTS] = "armyAttack_A2C_result"

RUN_TYPES[A3C] = {}
RUN_TYPES[A3C][DECISION_MAKER_TYPE] = "DecisionMakerOnlineAsync"
RUN_TYPES[A3C][TYPE] = "A3C"
RUN_TYPES[A3C][DIRECTORY] = "armyAttack_A3C"
RUN_TYPES[A3C][PARAMS] = A3C_PARAMS(STATE_SIZE, NUM_ACTIONS, numTrials2Learn=1, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[A3C][DECISION_MAKER_NAME] = "armyAttack_A3C"
RUN_TYPES[A3C][HISTORY] = "armyAttack_A3C_replayHistory"
RUN_TYPES[A3C][RESULTS] = "armyAttack_A3C_result"

RUN_TYPES[DQN_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DIRECTORY] = "armyAttack_dqn_Embedding"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(STATE_SIZE, STATE_END_ENEMY_MAT, NUM_ACTIONS, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DECISION_MAKER_NAME] = "armyAttack_dqn_Embedding"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][HISTORY] = "armyAttack_dqn_Embedding_replayHistory"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][RESULTS] = "armyAttack_dqn_Embedding_result"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "armyAttack_heuristic"
RUN_TYPES[NAIVE][RESULTS] = "armyAttack_heuristic"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"
    
STEP_DURATION = 0

def GetRunTypeArmyAttack(configDict):
    return RUN_TYPES[configDict[AGENT_NAME]]

def CreateDecisionMakerArmyAttack(configDict, isMultiThreaded, dmCopy=None, hyperParamsDict=None):
    dmCopy = "" if dmCopy==None else "_" + str(dmCopy)
    
    if configDict[AGENT_NAME] == "none":
        return BaseDecisionMaker(AGENT_NAME)
        
    runType = GetRunTypeArmyAttack(configDict)
    
    # create agent dir
    directory = configDict["directory"] + "/" + AGENT_DIR
    if not os.path.isdir("./" + directory):
        os.makedirs("./" + directory)
    directory += runType[DIRECTORY] + dmCopy

    if configDict[AGENT_NAME] == "naive":
        decisionMaker = NaiveDecisionMakerArmyAttack(NUM_TRIALS_2_SAVE, agentName=AGENT_NAME, resultFName=runType[RESULTS], directory=directory)
    else:
        dmClass = eval(runType[DECISION_MAKER_TYPE])
        if "learningRate" in configDict:
            runType[PARAMS].learning_rate = configDict["learningRate"]
        
        if hyperParamsDict != None:
            runType[PARAMS] = ChangeParamsAccordingToDict(runType[PARAMS], hyperParamsDict)
        elif "hyperParams" in configDict:
            runType[PARAMS] = ChangeParamsAccordingToDict(runType[PARAMS], configDict["hyperParams"])

        decisionMaker = dmClass(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], agentName=AGENT_NAME,
                                        resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory, isMultiThreaded=isMultiThreaded)
                                    

    return decisionMaker, runType

class SharedDataArmyAttack(EmptySharedData):
    def __init__(self):
        super(SharedDataArmyAttack, self).__init__()
        self.enemyArmyMat = [0] * (GRID_SIZE * GRID_SIZE)

class NaiveDecisionMakerArmyAttack(BaseNaiveDecisionMaker):
    def __init__(self, numTrials2Save, agentName = "", resultFName = None, directory = None):
        super(NaiveDecisionMakerArmyAttack, self).__init__(numTrials2Save, agentName=agentName, resultFName=resultFName, directory=directory)
        

    def choose_action(self, state, validActions, targetValues=False):
        if len(validActions) > 1:
            if np.random.uniform() > 0.9:
                return np.random.choice(validActions)

        return ACTION_DO_NOTHING

    def ActionsValues(self, state, validActions, target = True):
        vals = np.zeros(NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals

class ArmyAttack(BaseAgent):
    def __init__(self, sharedData, configDict, decisionMaker, isMultiThreaded, playList, trainList, testList, dmCopy=None):        
        super(ArmyAttack, self).__init__(STATE_SIZE)

        self.sharedData = sharedData

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker, _ = CreateDecisionMakerArmyAttack(configDict, isMultiThreaded, dmCopy=dmCopy)

        self.history = self.decisionMaker.AddHistory()
        # state and actions:

        self.terminalState = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        
        self.lastValidAttackAction = None
        self.enemyArmyGridLoc2ScreenLoc = {}

        self.rewardTarget = 0.0
        self.rewardNormal = 0.0

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
        super(ArmyAttack, self).FirstStep()

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.selfLocCoord = None

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)
        elif self.testAgent:
            self.decisionMaker.end_test_run(reward, score, stepNum)

    def Learn(self, reward, terminal):
        if self.trainAgent or self.testAgent:        
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

        if a > ACTION_DO_NOTHING:     
            goTo = self.enemyArmyGridLoc2ScreenLoc[a - ACTIONS_START_IDX_ATTACK].copy()
            if SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
       
        self.isActionCommitted = True
        self.lastActionCommitted = a

        return sc2Action, True

    def ChooseAction(self):

        if self.playAgent:
            validActions = self.ValidActions(self.current_scaled_state)
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)
        else:
            action = ACTION_DO_NOTHING

        self.current_action = action
        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyArmyLoc(obs)

        self.current_state[STATE_TIME_LINE_IDX] = self.sharedData.numStep

        for idx in range(GRID_SIZE * GRID_SIZE):
            self.sharedData.enemyArmyMat[idx] = self.current_state[STATE_START_ENEMY_MAT + idx]

        self.ScaleState()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state

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

    def GetEnemyArmyLoc(self, obs):
        playerType = obs.observation["feature_screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["feature_screen"][SC2_Params.UNIT_TYPE]

        enemyPoints = []
        enemyPower = []
        for unit in TerranUnit.ARMY:
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            
            if len(enemyArmy_y) > 0:
                if unit in TerranUnit.ARMY_SPEC.keys():
                    numScreenPixels = TerranUnit.ARMY_SPEC[unit].numScreenPixels
                else:
                    numScreenPixels = TerranUnit.DEFAULT_UNIT_NUM_SCREEN_PIXELS

                unitPoints, unitPower = CenterPoints(enemyArmy_y, enemyArmy_x, numScreenPixels)
                enemyPoints += unitPoints
                enemyPower += unitPower
            
        self.enemyArmyGridLoc2ScreenLoc = {}
        for i in range(len(enemyPoints)):
            idx = self.GetScaledIdx(enemyPoints[i])
            if idx in self.enemyArmyGridLoc2ScreenLoc.keys():
                self.current_state[STATE_START_ENEMY_MAT + idx] += enemyPower[i]
                self.enemyArmyGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyArmyGridLoc2ScreenLoc[idx], enemyPoints[i])
            else:
                self.current_state[STATE_START_ENEMY_MAT + idx] = enemyPower[i]
                self.enemyArmyGridLoc2ScreenLoc[idx] = enemyPoints[i]     

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
        if self.illigalmoveSolveInModel:
            valid = [ACTION_DO_NOTHING]
            enemiesLoc = (state[STATE_START_ENEMY_MAT:STATE_END_ENEMY_MAT] > 0).nonzero()
            for loc in enemiesLoc[0]:
                valid.append(loc + ACTIONS_START_IDX_ATTACK)

            return valid
        else:
            return list(range(NUM_ACTIONS))

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



if __name__ == "__main__":
    from absl import app
    from absl import flags
    flags.DEFINE_string("directoryPrefix", "", "directory names to take results")
    flags.DEFINE_string("directoryNames", "", "directory names to take results")
    flags.DEFINE_string("grouping", "100", "grouping size of results.")
    flags.DEFINE_string("max2Plot", "none", "grouping size of results.")
    flags.FLAGS(sys.argv)

    directoryNames = (flags.FLAGS.directoryNames).split(",")
    for d in range(len(directoryNames)):
        directoryNames[d] = flags.FLAGS.directoryPrefix + directoryNames[d]
    
    grouping = int(flags.FLAGS.grouping)
    if flags.FLAGS.max2Plot == "none":
        max2Plot = None
    else:
        max2Plot = int(flags.FLAGS.max2Plot)

    if "results" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot)
    elif "multipleResults" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot, multipleDm=True)

