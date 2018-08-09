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

from utils import BaseAgent

import tensorflow as tf

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay
from utils_decisionMaker import BaseDecisionMaker

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

from utils_results import PlotResults

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints

AGENT_DIR = "ArmyAttack/"
AGENT_NAME = "army_attack"


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

TIME_LINE_BUCKETING = 25

# possible types of decision maker

QTABLE = 'q'
DQN = 'dqn'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding' 

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
RUN_TYPES[QTABLE][DIRECTORY] = "armyAttack_q"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "armyAttack_q_qtable"
RUN_TYPES[QTABLE][HISTORY] = "armyAttack_q_replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "armyAttack_q_result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][DIRECTORY] = "armyAttack_dqn"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "armyAttack_dqn"
RUN_TYPES[DQN][HISTORY] = "armyAttack_dqn_replayHistory"
RUN_TYPES[DQN][RESULTS] = "armyAttack_dqn_result"

RUN_TYPES[DQN_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DIRECTORY] = "armyAttack_dqn_Embedding"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(STATE_SIZE, STATE_END_ENEMY_MAT, NUM_ACTIONS)
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][DECISION_MAKER_NAME] = "armyAttack_dqn_Embedding"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][HISTORY] = "armyAttack_dqn_Embedding_replayHistory"
RUN_TYPES[DQN_EMBEDDING_LOCATIONS][RESULTS] = "armyAttack_dqn_Embedding_result"

RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][TYPE] = "play"
    
STEP_DURATION = 0

class NaiveDecisionMakerArmyAttack(BaseDecisionMaker):
    def __init__(self):
        super(NaiveDecisionMakerArmyAttack, self).__init__()
        

    def choose_action(self, observation):
        armyPnts = (observation[STATE_START_ENEMY_MAT:STATE_END_ENEMY_MAT] > 0).nonzero()[0]
        selfLocs = (observation[STATE_START_SELF_MAT:STATE_END_SELF_MAT] > 0).nonzero()[0]

        if len(selfLocs) == 0 or len(armyPnts) == 0:
            return ACTION_DO_NOTHING

        self_y = int(selfLocs[0] / GRID_SIZE)
        self_x = selfLocs[0] % GRID_SIZE

        minDist = 1000
        minIdx = -1

        for idx in armyPnts:
            p_y = int(idx / GRID_SIZE)
            p_x = idx % GRID_SIZE
            diffX = p_x - self_x
            diffY = p_y - self_y

            dist = diffX * diffX + diffY * diffY
            if dist < minDist:
                minDist = dist
                minIdx = idx

        return minIdx + ACTIONS_START_IDX_ATTACK

    def ActionValuesVec(self, state, target = True):
        vals = np.zeros(NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state)] = 1.0

        return vals

class ArmyAttack(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):        
        super(ArmyAttack, self).__init__()

        self.sharedData = sharedData

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        # state and actions:

        self.terminalState = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        
        self.lastValidAttackAction = None
        self.enemyArmyGridLoc2ScreenLoc = {}

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        
        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerArmyAttack()
        else:
            runType = RUN_TYPES[dmTypes[AGENT_NAME]]

            # create agent dir
            directory = dmTypes["directory"] + "/" + AGENT_DIR
            if not os.path.isdir("./" + directory):
                os.makedirs("./" + directory)
            decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=AGENT_DIR+runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1

    def step(self, obs, moveNum):
        super(ArmyAttack, self).step(obs)
        if obs.first():
            self.FirstStep(obs)
               
        if moveNum == 0:
            self.CreateState(obs)
            self.Learn()
            self.current_action = self.ChooseAction()


        self.numStep += 1


        return self.current_action

    def FirstStep(self, obs):
        self.numStep = 0

        self.current_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
        
        self.current_action = None
        self.enemyArmyGridLoc2ScreenLoc = {}
        self.selfLocCoord = None      
        self.errorOccur = False

        self.prevReward = 0

    def LastStep(self, obs, reward = 0):
        if self.trainAgent:
            if reward == 0:
                if obs.reward > 0:
                    reward = 1.0
                else:
                    reward = -1.0

            if self.current_action is not None:
                self.decisionMaker.learn(self.current_state.copy(), self.current_action, float(reward), self.terminalState.copy(), True)

            score = obs.observation["score_cumulative"][0]
            self.decisionMaker.end_run(reward, score, self.numStep)
    
    def Learn(self, reward = 0):
        if self.trainAgent and self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, float(self.prevReward), self.current_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.prevReward = 0.0

    def IsDoNothingAction(self, a):
        return a == ACTION_DO_NOTHING

    def Action2Str(self, a):
        return ACTION2STR[a]

    def Action2SC2Action(self, obs, a, moveNum):
        if SC2_Actions.STOP in obs.observation['available_actions']:
            sc2Action = SC2_Actions.STOP_SC2_ACTION
        else:
            sc2Action = SC2_Actions.DO_NOTHING_SC2_ACTION

        if self.current_action > ACTION_DO_NOTHING:     
            goTo = self.enemyArmyGridLoc2ScreenLoc[self.current_action - ACTIONS_START_IDX_ATTACK].copy()
            if SC2_Actions.ATTACK_SCREEN in obs.observation['available_actions']:
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_SCREEN, [SC2_Params.NOT_QUEUED, SwapPnt(goTo)])
       
        return sc2Action, True

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
            action = ACTION_DO_NOTHING

        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyArmyLoc(obs)
        self.current_state[STATE_TIME_LINE_IDX] = int(self.numStep / TIME_LINE_BUCKETING)

    def GetSelfLoc(self, obs):
        playerType = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["screen"][SC2_Params.UNIT_TYPE]

        allArmy_y = []
        allArmy_x = [] 
        for key, spec in TerranUnit.UNIT_SPEC.items():
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
        playerType = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["screen"][SC2_Params.UNIT_TYPE]

        enemyPoints = []
        enemyPower = []
        for unit, spec in TerranUnit.UNIT_SPEC.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            unitPoints, unitPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
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
        
    def ValidActions(self):
        valid = [ACTION_DO_NOTHING]
        for key in self.enemyArmyGridLoc2ScreenLoc.keys():
            valid.append(key + ACTIONS_START_IDX_ATTACK)

        return valid

    def PrintState(self):
        print("\n\nstate: timeline =", self.current_state[STATE_TIME_LINE_IDX], "last attack action =", self.lastValidAttackAction)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx = STATE_START_SELF_MAT + x + y * GRID_SIZE
                print(int(self.current_state[idx]), end = '')
            
            print(end = '  |  ')
            
            for x in range(GRID_SIZE):
                idx = STATE_START_ENEMY_MAT + x + y * GRID_SIZE
                print(int(self.current_state[idx]), end = '')

            print('||')


if __name__ == "__main__":
    if "results" in sys.argv:
        PlotResults(AGENT_NAME, RUN_TYPES)
        # configFileNames = []
        # for arg in sys.argv:
        #     if arg.find(".txt") >= 0:
        #         configFileNames.append(arg)

        # configFileNames.sort()
        # resultFnames = []
        # directoryNames = []
        # for configFname in configFileNames:
        #     dm_Types = eval(open(configFname, "r+").read())
        #     runType = RUN_TYPES[dm_Types[AGENT_NAME]]
            
        #     directory = dm_Types["directory"]
        #     fName = runType[RESULTS]
            
        #     if DIRECTORY in runType.keys():
        #         dirName = runType[DIRECTORY]
        #     else:
        #         dirName = ''

        #     resultFnames.append(fName)
        #     directoryNames.append(directory + "/" + AGENT_DIR + dirName)

        # grouping = int(sys.argv[len(sys.argv) - 1])
        # plot = PlotMngr(resultFnames, directoryNames, configFileNames, AGENT_DIR)
        # plot.Plot(grouping)