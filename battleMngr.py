# build base sub agent
import sys
import random
import math
import time
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.lib import actions


#sub-agents
from baseAttack import BaseAttack
from armyAttack import ArmyAttack

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

#decision makers
from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay

from utils_results import ResultFile
from utils_results import PlotMngr

# params
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_EMBEDDING_PARAMS
from utils_qtable import QTableParams
from utils_qtable import QTableParamsExplorationDecay

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints

TIME_LINE_BUCKETING = 25
NON_VALID_LOCATION_QVAL = -2

def NumActions(gridSize):
    return gridSize * gridSize + 1
def NumStateLocationsVal(gridSize):
    return 3 * gridSize * gridSize
def NumStateVal(gridSize):
    return 3 * gridSize * gridSize + 1


def NNFunc_2Layers(x, numActions, scope):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        output = tf.contrib.layers.fully_connected(fc2, numActions, activation_fn = tf.nn.sigmoid) * 2 - 1
    return output

#physical actions from q table
ACTION_DO_NOTHING = 0
ACTION_NORTH = 1
ACTION_SOUTH = 2
ACTION_EAST = 3
ACTION_WEST = 4
ACTION_NORTH_EAST = 5
ACTION_NORTH_WEST = 6
ACTION_SOUTH_EAST = 7
ACTION_SOUTH_WEST = 8

NUM_PHYSICAL_ACTIONS =  9

NUM_UNIT_SCREEN_PIXELS = 0

SCREEN_MIN = [3,3]
SCREEN_MAX = [59,80]

for key,value in TerranUnit.UNIT_SPEC.items():
    if value.name == "marine":
        NUM_UNIT_SCREEN_PIXELS = value.numScreenPixels

# possible types of play

QTABLE_GS5 = 'qGS5'
DQN_GS5_ARMY = 'dqnGS5'
DQN_GS5_EMBEDDING_LOCATIONS = 'dqnGS5_Embedding' 
DQN_GS5_2LAYERS = 'dqnGS5_2Layers' 

NAIVE_DECISION = 'naive'

USER_PLAY = 'play'

ALL_TYPES = set([USER_PLAY, QTABLE_GS5, DQN_GS5_ARMY, DQN_GS5_EMBEDDING_LOCATIONS, DQN_GS5_2LAYERS, NAIVE_DECISION])

def BaseTestSet(gridSize):    
    testSet = []

    s1 = np.zeros(3 * gridSize * gridSize + 1, dtype = int)
    
    s1[8] = 5
    s1[gridSize * gridSize + 8] = 1
    s1[gridSize * gridSize + 9] = 1
    s1[gridSize * gridSize + 13] = 1
    s1[2 * gridSize * gridSize + 14] = 1
    actions_s1 = [8, 9, 13, 14, gridSize * gridSize]

    testSet.append([s1, actions_s1])

    s2 = np.zeros(3 * gridSize * gridSize + 1, dtype = int)

    s2[5] = 2
    s2[gridSize * gridSize + 8] = 2
    s2[gridSize * gridSize + 14] = 3
    actions_s2 = [8, 14, gridSize * gridSize]

    testSet.append([s2, actions_s2])

    s3 = np.zeros(3 * gridSize * gridSize + 1, dtype = int)
    s3[11] = 5
    s3[gridSize * gridSize + 12] = 2
    s3[gridSize * gridSize + 23] = 2
    s3[2 * gridSize * gridSize + 13] = 1
    actions_s3 = [12, 13, 23, gridSize * gridSize]

    testSet.append([s3, actions_s3])

    return testSet

# data for run type
TYPE = "type"
NN = "nn"
Q_TABLE_INPUT = "q_using"
Q_TABLE = "q"
T_TABLE = "t"
HISTORY = "hist"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'
GRIDSIZE_key = 'gridsize'
BUILDING_VALUES_key = 'buildingValues'
DIRECTORY = 'directory'

# table names
RUN_TYPES = {}


RUN_TYPES[QTABLE_GS5] = {}
RUN_TYPES[QTABLE_GS5][TYPE] = "QReplay"
RUN_TYPES[QTABLE_GS5][GRIDSIZE_key] = 5
RUN_TYPES[QTABLE_GS5][PARAMS] = QTableParamsExplorationDecay(NumStateVal(5), NumActions(5), states2Monitor = BaseTestSet(5))
RUN_TYPES[QTABLE_GS5][NN] = ""
RUN_TYPES[QTABLE_GS5][Q_TABLE] = "battleMngr_qGS5_qtable"
RUN_TYPES[QTABLE_GS5][HISTORY] = "battleMngr_qGS5_replayHistory"
RUN_TYPES[QTABLE_GS5][RESULTS] = "battleMngr_qGS5_result"

RUN_TYPES[DQN_GS5_ARMY] = {}
RUN_TYPES[DQN_GS5_ARMY][TYPE] = "NN"
RUN_TYPES[DQN_GS5_ARMY][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_ARMY][PARAMS] = DQN_PARAMS(NumStateVal(5), 3, states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_ARMY][NN] = "battleMngr_dqnGS5_DQN"
RUN_TYPES[DQN_GS5_ARMY][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_ARMY][HISTORY] = "battleMngr_dqnGS5_replayHistory"
RUN_TYPES[DQN_GS5_ARMY][RESULTS] = "battleMngr_dqnGS5_result"

RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS] = {}
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][TYPE] = "NN"
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][PARAMS] = DQN_EMBEDDING_PARAMS(NumStateVal(5), NumStateLocationsVal(5), 3, states2Monitor = BaseTestSet(5))
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][NN] = "battleMngr_dqnGS5_Embedding_DQN"
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][HISTORY] = "battleMngr_dqnGS5_Embedding_replayHistory"
RUN_TYPES[DQN_GS5_EMBEDDING_LOCATIONS][RESULTS] = "battleMngr_dqnGS5_Embedding_result"

RUN_TYPES[DQN_GS5_2LAYERS] = {}
RUN_TYPES[DQN_GS5_2LAYERS][TYPE] = "NN"
RUN_TYPES[DQN_GS5_2LAYERS][GRIDSIZE_key] = 5
RUN_TYPES[DQN_GS5_2LAYERS][PARAMS] = DQN_PARAMS(NumStateVal(5), 3, states2Monitor = BaseTestSet(5), nn_Func = NNFunc_2Layers)
RUN_TYPES[DQN_GS5_2LAYERS][NN] = "battleMngr_dqnGS5_2Layers_DQN"
RUN_TYPES[DQN_GS5_2LAYERS][Q_TABLE] = ""
RUN_TYPES[DQN_GS5_2LAYERS][HISTORY] = "battleMngr_dqnGS5_2Layers_replayHistory"
RUN_TYPES[DQN_GS5_2LAYERS][RESULTS] = "battleMngr_dqnGS5_2Layers_result"


RUN_TYPES[USER_PLAY] = {}
RUN_TYPES[USER_PLAY][GRIDSIZE_key] = 5
RUN_TYPES[USER_PLAY][TYPE] = "play"

RUN_TYPES[NAIVE_DECISION] = {}
RUN_TYPES[NAIVE_DECISION][GRIDSIZE_key] = 5
RUN_TYPES[NAIVE_DECISION][RESULTS] = "battleMngr_naive_result"
RUN_TYPES[NAIVE_DECISION][TYPE] = "naive"

class NaiveDecisionMakerMngr:
    def __init__(self, gridSize, resultFName, armyAttackAction = 1, buildingAttackAction = 2, doNothingAction = 0):
        self.resultsFile = ResultFile(resultFName)
        
        self.gridSize = gridSize
        self.startEnemyMat = gridSize * gridSize
        self.startBuildingMat = 2 * gridSize * gridSize
        self.endBuildingMat = 3 * gridSize * gridSize

        self.armyAttackAction = armyAttackAction
        self.buildingAttackAction = buildingAttackAction
        self.doNothingAction = doNothingAction
        self.numActions = 3

    def choose_action(self, observation):
        if (observation[self.startEnemyMat:self.startBuildingMat] > 0).any():
            return self.armyAttackAction
        elif (observation[self.startBuildingMat:self.endBuildingMat] > 0).any():
            return self.buildingAttackAction
        else:
            return self.doNothingAction

    def learn(self, s, a, r, s_, terminal = False):
        # if r != 0:
        #     print(r)
        return None
    def actionValuesVec(self,state):
        return np.zeros(self.numActions,dtype = float)
    def end_run(self, r, score = 0 ,steps = 0):
        self.resultsFile.end_run(r,score,steps, True)
        return True
    def ExploreProb(self):
        return 0


STEP_DURATION = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])
STOP_SC2_ACTION = actions.FunctionCall(SC2_Actions.STOP, [SC2_Params.NOT_QUEUED])

TRAIN_POSSIBIITIES = ["mngr", "army", "base"]
class Attack(base_agent.BaseAgent):
    def __init__(self):        
        super(Attack, self).__init__()   

        runTypeArg = ALL_TYPES.intersection(sys.argv)
        if len(runTypeArg) != 1:
            print("\n\nplay type not entered correctly\n\n")
            exit() 

        runArg = runTypeArg.pop()

        if "mngr" in sys.argv:
            self.battleMngr = Mngr(True, runArg)
            self.isMngrTrain = True
            self.mainAgent = self.battleMngr
        else:
            self.battleMngr = Mngr(False, runArg)
            self.isMngrTrain = False


        #sess = self.battleMngr.decisionMaker.dqn.sess
        if "army" in sys.argv:
            self.armyAttack = ArmyAttack(True, runArg)
            self.mainAgent = self.armyAttack
        else:
            self.armyAttack = ArmyAttack(False, "dqnGS5_Embedding")
            #self.armyAttack = ArmyAttack(False, runArg)

        if "base" in sys.argv: 
            self.baseAttack = BaseAttack(True, runArg)
            self.mainAgent = self.baseAttack
        else:
            self.baseAttack = BaseAttack(False, "dqnGS5_Embedding")
            #self.baseAttack = BaseAttack(False, runArg)

    def step(self, obs):
        super(Attack, self).step(obs)
        if self.isMngrTrain:
            if obs.first():
                self.baseAttack.FirstStep(obs)
                self.armyAttack.FirstStep(obs)
            
            a = self.mainAgent.step(obs)
            return self.Action2SC2Action(obs,a)
        else:
            return self.mainAgent.step(obs)

    def Action2SC2Action(self, obs, a):
        if a == self.battleMngr.action_BaseAttack:
            subAction = self.baseAttack.step(obs)
            sc2Action = self.baseAttack.Action2SC2Action(obs, subAction)
        elif a == self.battleMngr.action_ArmyAttack:
            subAction = self.armyAttack.step(obs)
            sc2Action = self.armyAttack.Action2SC2Action(obs, subAction)
        else:
            if SC2_Actions.STOP in obs.observation['available_actions']:
                sc2Action = STOP_SC2_ACTION
            else:
                sc2Action = DO_NOTHING_SC2_ACTION
        
        return sc2Action



class Mngr(base_agent.BaseAgent):
    def __init__(self, mainAgent = True, runArg = '', tfSession = None):        
        super(Mngr, self).__init__()

        self.mainAgent = mainAgent

        self.illigalmoveSolveInModel = True

        if runArg == '':
            runTypeArg = ALL_TYPES.intersection(sys.argv)
            if len(runTypeArg) != 1:
                print("\n\nplay type not entered correctly\n\n")
                exit() 

            runArg = runTypeArg.pop()

        runType = RUN_TYPES[runArg]

        self.current_action = None
        self.armyExist = True
        self.buildingsExist = True
        # state and actions:

        self.gridSize = runType[GRIDSIZE_key]
        self.numActions = 3
        self.action_DoNothing = 0
        self.action_ArmyAttack = 1
        self.action_BaseAttack = 2

        self.state_startSelfMat = 0
        self.state_startEnemyMat = self.gridSize * self.gridSize
        self.state_startBuildingMat = 2 * self.gridSize * self.gridSize
        self.state_timeLineIdx = 3 * self.gridSize * self.gridSize

        self.state_size = 3 * self.gridSize * self.gridSize + 1
        self.terminalState = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.BuildingValues = {}
        if BUILDING_VALUES_key in runType.keys():
            for spec in TerranUnit.BUILDING_SPEC.values():
                self.BuildingValues[spec.name] = spec.buildingValues
        else:
            for spec in TerranUnit.BUILDING_SPEC.values():
                self.BuildingValues[spec.name] = 1

        # create decision maker
        if runType[TYPE] == 'NN' or runType[TYPE] == 'QReplay':
            runType[PARAMS].stateSize = self.state_size
            self.decisionMaker = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])     
        elif runType[TYPE] == 'play':
            self.decisionMaker = UserPlay(False)
        elif runType[TYPE] == 'naive':
            self.illigalmoveSolveInModel = False
            self.decisionMaker = NaiveDecisionMakerMngr(self.gridSize, runType[RESULTS])
        else:
            print("\n\ninvalid run type\n\n")
            exit()

    def step(self, obs):
        super(Mngr, self).step(obs)   
        
        if obs.first():
            return self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
            

        self.CreateState(obs)
        if self.mainAgent:
            self.Learn()
            time.sleep(STEP_DURATION)
        
        self.current_action = self.ChooseAction()

        self.numStep += 1        
        time.sleep(STEP_DURATION)

        # print("\n")
        # self.PrintState()
        return self.current_action

    def FirstStep(self, obs):        
        self.numStep = 0

        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
        self.previous_state = np.zeros(self.state_size, dtype=np.int, order='C')
        
        self.current_action = None

        self.prevReward = 0

        return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])

    def LastStep(self, obs):
        if self.mainAgent:
            if obs.reward > 0:
                reward = 1
            elif obs.reward < 0:
                reward = -1
            else:
                reward = -1

            if self.current_action is not None:
                self.decisionMaker.learn(self.current_state.copy(), self.current_action, float(reward), self.terminalState.copy(), True)

            score = obs.observation["score_cumulative"][0]
            self.decisionMaker.end_run(reward, score, self.numStep)
    
    def Learn(self):
        if self.current_action is not None:
            self.decisionMaker.learn(self.previous_state.copy(), self.current_action, self.prevReward, self.current_state.copy())

        self.previous_state[:] = self.current_state[:]
        self.prevReward = 0

    def ChooseAction(self):
        if self.illigalmoveSolveInModel:
            validActions = self.ValidActions()
            if np.random.uniform() > self.decisionMaker.ExploreProb():
                valVec = self.decisionMaker.actionValuesVec(self.current_state)   
                random.shuffle(validActions)
                validVal = valVec[validActions]
                action = validActions[validVal.argmax()]
            else:
                action = np.random.choice(validActions) 
        else:
            action = self.decisionMaker.choose_action(self.current_state)

        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(self.state_size, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyArmyLoc(obs)
        self.GetEnemyBuildingLoc(obs)
        
        self.current_state[self.state_timeLineIdx] = int(self.numStep / TIME_LINE_BUCKETING)

    def GetSelfLoc(self, obs):
        screenMap = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        s_y, s_x = (screenMap == SC2_Params.PLAYER_SELF).nonzero()
        selfPoints, selfPower = CenterPoints(s_y, s_x)

        for i in range(len(selfPoints)):
            idx = self.GetScaledIdx(selfPoints[i])
            power = math.ceil(selfPower[i] /  NUM_UNIT_SCREEN_PIXELS)
            self.current_state[self.state_startSelfMat + idx] += power

        if len(s_y) > 0:
            self.selfLocCoord = [int(sum(s_y) / len(s_y)), int(sum(s_x) / len(s_x))]

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
        
        self.armyExist = False
        for i in range(len(enemyPoints)):
            self.armyExist = True
            idx = self.GetScaledIdx(enemyPoints[i])
            self.current_state[self.state_startEnemyMat + idx] += enemyPower[i]

    def GetEnemyBuildingLoc(self, obs):
        playerType = obs.observation["screen"][SC2_Params.PLAYER_RELATIVE]
        unitType = obs.observation["screen"][SC2_Params.UNIT_TYPE]

        enemyBuildingPoints = []
        enemyBuildingPower = []
        for unit, spec in TerranUnit.BUILDING_SPEC.items():
            enemyArmy_y, enemyArmy_x = ((unitType == unit) & (playerType == SC2_Params.PLAYER_HOSTILE)).nonzero()
            buildingPoints, buildingPower = CenterPoints(enemyArmy_y, enemyArmy_x, spec.numScreenPixels)
            enemyBuildingPoints += buildingPoints
            enemyBuildingPower += buildingPower * self.BuildingValues[spec.name]
        
        self.buildingsExist = False
        for i in range(len(enemyBuildingPoints)):
            self.buildingsExist = True
            idx = self.GetScaledIdx(enemyBuildingPoints[i])
            self.current_state[self.state_startBuildingMat + idx] += enemyBuildingPower[i]
     
       
    def GetScaledIdx(self, screenCord):
        locX = screenCord[SC2_Params.X_IDX]
        locY = screenCord[SC2_Params.Y_IDX]

        yScaled = int((locY / SC2_Params.SCREEN_SIZE) * self.gridSize)
        xScaled = int((locX / SC2_Params.SCREEN_SIZE) * self.gridSize)

        return xScaled + yScaled * self.gridSize
    
    def Closest2Self(self, p1, p2):
        d1 = DistForCmp(p1, self.selfLocCoord)
        d2 = DistForCmp(p2, self.selfLocCoord)
        if d1 < d2:
            return p1
        else:
            return p2
    
    def ValidActions(self):
        valid = [self.action_DoNothing]
        if self.armyExist:
            valid.append(self.action_ArmyAttack)
        if self.buildingsExist:
            valid.append(self.action_BaseAttack)
        
        return valid

    def PrintState(self):
        print("\n\nstate: timeline =", self.current_state[self.state_timeLineIdx])
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                idx = self.state_startSelfMat + x + y * self.gridSize
                print(int(self.current_state[idx]), end = '')
            
            print(end = '  |  ')
            
            for x in range(self.gridSize):
                idx = self.state_startEnemyMat + x + y * self.gridSize
                print(int(self.current_state[idx]), end = '')

            print(end = '  |  ')
            
            for x in range(self.gridSize):
                idx = self.state_startBuildingMat + x + y * self.gridSize
                if self.current_state[idx] < 10:
                    print(self.current_state[idx], end = '  ')
                else:
                    print(self.current_state[idx], end = ' ')

            print('||')


if __name__ == "__main__":
    if "plotResults" in sys.argv:
        runTypeArg = list(ALL_TYPES.intersection(sys.argv))
        runTypeArg.sort()
        resultFnames = []
        directoryNames = []
        for arg in runTypeArg:
            runType = RUN_TYPES[arg]
            fName = runType[RESULTS]
            
            if DIRECTORY in runType.keys():
                dirName = runType[DIRECTORY]
            else:
                dirName = ''

            resultFnames.append(fName)
            directoryNames.append(dirName)

        grouping = int(sys.argv[len(sys.argv) - 1])
        plot = PlotMngr(resultFnames, directoryNames, runTypeArg)
        plot.Plot(grouping)