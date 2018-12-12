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
from algo_decisionMaker import BaseNaiveDecisionMaker

from algo_decisionMaker import CreateDecisionMaker

from utils_results import PlotResults

from utils import SwapPnt
from utils import DistForCmp
from utils import CenterPoints

AGENT_NAME = "army_attack"

NUM_TRIALS_2_SAVE = 100

GRID_SIZE = 5

class ArmyAttackActions:
    DO_NOTHING = 0
    START_IDX_ATTACK = 1
    END_IDX_ATTACK = START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
    SIZE = END_IDX_ATTACK

ACTION2STR = {}

ACTION2STR[ArmyAttackActions.DO_NOTHING] = "Do_Nothing"
for a in range(ArmyAttackActions.START_IDX_ATTACK, ArmyAttackActions.END_IDX_ATTACK):
    ACTION2STR[a] = "ArmyAttack_" + str(a - ArmyAttackActions.START_IDX_ATTACK)

class ArmyAttackState:
    START_SELF_MAT = 0
    END_SELF_MAT = START_SELF_MAT + GRID_SIZE * GRID_SIZE
    START_ENEMY_MAT = END_SELF_MAT
    END_ENEMY_MAT = START_ENEMY_MAT + GRID_SIZE * GRID_SIZE
    TIME_LINE_IDX = END_ENEMY_MAT
    SIZE = TIME_LINE_IDX + 1
 
STEP_DURATION = 0

class SharedDataArmyAttack(EmptySharedData):
    def __init__(self):
        super(SharedDataArmyAttack, self).__init__()
        self.enemyArmyMat = [0] * (GRID_SIZE * GRID_SIZE)

class StochasticHeuristicArmyAttackDm(BaseNaiveDecisionMaker):
    def __init__(self, numTrials2Save=None, agentName="", resultFName=None, directory=None):
        super(StochasticHeuristicArmyAttackDm, self).__init__(numTrials2Save, agentName=agentName, resultFName=resultFName, directory=directory)
        

    def choose_action(self, state, validActions, targetValues=False):
        if len(validActions) > 1:
            if np.random.uniform() > 0.9:
                return np.random.choice(validActions)

        return ArmyAttackActions.DO_NOTHING

    def ActionsValues(self, state, validActions, target = True):
        vals = np.zeros(ArmyAttackActions.SIZE,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals
    
class AttackClosestArmyAttackDm(BaseNaiveDecisionMaker):
    def __init__(self, numTrials2Save=None, agentName="", resultFName=None, directory=None):
        super(AttackClosestArmyAttackDm, self).__init__(numTrials2Save, agentName=agentName, resultFName=resultFName, directory=directory)
        

    def choose_action(self, state, validActions, targetValues=False):
        if len(validActions) > 1:
            selfLocations = []
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    idx = x + y * GRID_SIZE
                    if state[ArmyAttackState.START_SELF_MAT + idx]:
                        selfLocations.append([y, x])
            
            if len(selfLocations) > 1:
                avgSelfLocation = np.average(selfLocations, axis=0)
            elif len(selfLocations) > 0:
                avgSelfLocation = selfLocations[0]
            else:
                return ArmyAttackActions.DO_NOTHING
                
            activeActions = validActions.copy()
            activeActions.remove(ArmyAttackActions.DO_NOTHING)
            
            minDist = 2 * GRID_SIZE ** 2
            minDistAction = -1
            for a in activeActions:
                idxOnMap = a - ArmyAttackActions.START_IDX_ATTACK
                x = idxOnMap % GRID_SIZE
                y = int(idxOnMap / GRID_SIZE)
                dist = (y - avgSelfLocation[0]) ** 2 + (x - avgSelfLocation[1]) ** 2
                if dist < minDist:
                    minDist = dist
                    minDistAction = a

            if minDistAction in activeActions:
                return minDistAction
            else:
                print("ERROR")
            

        return ArmyAttackActions.DO_NOTHING

    def ActionsValues(self, state, validActions, target = True):
        vals = np.zeros(ArmyAttackActions.SIZE,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals

class ArmyAttack(BaseAgent):
    def __init__(self, sharedData, configDict, decisionMaker, isMultiThreaded, playList, trainList, testList, dmCopy=None):        
        super(ArmyAttack, self).__init__(ArmyAttackState.SIZE)

        self.sharedData = sharedData

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        self.trainAgent = AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker, _ = CreateDecisionMaker(agentName=AGENT_NAME, configDict=configDict, 
                            isMultiThreaded=isMultiThreaded, dmCopy=dmCopy, heuristicClass=AttackClosestArmyAttackDm)

        self.history = self.decisionMaker.AddHistory()
        # state and actions:

        self.terminalState = np.zeros(ArmyAttackState.SIZE, dtype=np.int, order='C')
        
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

        self.current_state = np.zeros(ArmyAttackState.SIZE, dtype=np.int, order='C')
        self.current_scaled_state = np.zeros(ArmyAttackState.SIZE, dtype=np.int, order='C')
        self.previous_scaled_state = np.zeros(ArmyAttackState.SIZE, dtype=np.int, order='C')
        
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
                for a in range(ArmyAttackActions.SIZE):
                    self.history.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.history.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def IsDoNothingAction(self, a):
        return a == ArmyAttackActions.DO_NOTHING

    def Action2Str(self, a, onlyAgent=False):
        return ACTION2STR[a]

    def Action2SC2Action(self, obs, a, moveNum):
        if SC2_Actions.STOP in obs.observation['available_actions']:
            sc2Action = SC2_Actions.STOP_SC2_ACTION
        else:
            sc2Action = SC2_Actions.DO_NOTHING_SC2_ACTION

        if a > ArmyAttackActions.DO_NOTHING:     
            goTo = self.enemyArmyGridLoc2ScreenLoc[a - ArmyAttackActions.START_IDX_ATTACK].copy()
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
            action = ArmyAttackActions.DO_NOTHING

        self.current_action = action
        return action

    def CreateState(self, obs):
        self.current_state = np.zeros(ArmyAttackState.SIZE, dtype=np.int, order='C')
    
        self.GetSelfLoc(obs)
        self.GetEnemyArmyLoc(obs)

        self.current_state[ArmyAttackState.TIME_LINE_IDX] = self.sharedData.numStep

        for idx in range(GRID_SIZE * GRID_SIZE):
            self.sharedData.enemyArmyMat[idx] = self.current_state[ArmyAttackState.START_ENEMY_MAT + idx]

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
                self.current_state[ArmyAttackState.START_SELF_MAT + idx] += power

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
                self.current_state[ArmyAttackState.START_ENEMY_MAT + idx] += enemyPower[i]
                self.enemyArmyGridLoc2ScreenLoc[idx] = self.Closest2Self(self.enemyArmyGridLoc2ScreenLoc[idx], enemyPoints[i])
            else:
                self.current_state[ArmyAttackState.START_ENEMY_MAT + idx] = enemyPower[i]
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
        
        return p1 if d1 < d2 else p2
        
    def ValidActions(self, state):
        if self.illigalmoveSolveInModel:
            valid = [ArmyAttackActions.DO_NOTHING]
            enemiesLoc = (state[ArmyAttackState.START_ENEMY_MAT:ArmyAttackState.END_ENEMY_MAT] > 0).nonzero()
            for loc in enemiesLoc[0]:
                valid.append(loc + ArmyAttackActions.START_IDX_ATTACK)

            return valid
        else:
            return list(range(ArmyAttackActions.SIZE))

    def PrintState(self):
        print("\n\nstate: timeline =", self.current_scaled_state[ArmyAttackState.TIME_LINE_IDX], "last attack action =", self.lastValidAttackAction)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx = ArmyAttackState.START_SELF_MAT + x + y * GRID_SIZE
                print(int(self.current_scaled_state[idx]), end = '')
            
            print(end = '  |  ')
            
            for x in range(GRID_SIZE):
                idx = ArmyAttackState.START_ENEMY_MAT + x + y * GRID_SIZE
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
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot)
    elif "multipleResults" in sys.argv:
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot, multipleDm=True)

