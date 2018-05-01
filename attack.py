# build base sub agent
import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.lib import actions

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions
from utils import QLearningTable

from utils import SwapPnt
from utils import FindMiddle
from utils import GetScreenCorners
from utils import CreateCameraHeightMap
from utils import HaveSpace
from utils import IsolateArea
from utils import GetCoord
from utils import PowerSurroundPnt
from utils import BattleStarted


ID_ACTION_DO_NOTHING = 0
ID_ACTION_ATTACK_NEAREST = 1
ID_ACTION_ATTACK_NEAREST_WITH_POWER = 2
ID_ACTION_ATTACK_STRONGEST = 3

NUM_ACTIONS = 4

# state details
STATE_NON_VALID_NUM = -1
STATE_GRID_SIZE = 2
STATE_POWER_BUCKETING = 10
STATE_SELF_ARMY_BUCKETING = 5

STATE_ARMY_POWER_IDX = 0
STATE_ENEMY_BASE_POWER_IDX = 3
STATE_ENEMY_ARMY_POWER_IDX = 1
STATE_ENEMY_ARMY_LOCATION_IDX = 2

STATE_SIZE = 5

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

class AttackCmd:
    def __init__(self, state, attackCoord, isBaseAttack ,attackStarted = False):
        self.m_state = state
        self.m_attackCoord = [math.ceil(attackCoord[0]), math.ceil(attackCoord[1])]
        self.m_inTheWayBattle = [-1, -1]
        self.m_isBaseAttack = isBaseAttack
        self.m_attackStarted = attackStarted
        self.m_attackEnded = False

def ScaleLoc2Grid(x, y):
    x /= (SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX] / STATE_GRID_SIZE)
    y /= (SC2_Params.MINIMAP_SIZE[SC2_Params.Y_IDX] / STATE_GRID_SIZE)
    return int(x) + int(y) * STATE_GRID_SIZE

class AttackSubAgent:
    def __init__(self, qTableName):        

        # states and action:
        self.current_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        # model params
        self.unit_type = None

        self.baseTopLeft = False
        self.enemyBaseCoord = [-1,-1]
        self.enemyBaseDestroyed = False

        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]

        self.currentBuildingTypeSelected = TerranUnit.BARRACKS
        self.currentBuildingCoordinate = [-1,-1]

        self.attackCommands = []


    def step(self, obs):
        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]

        sc2Action = DO_NOTHING_SC2_ACTION

        if self.move_number == 0:
            # and len (self.attackCommands) == 0
            if SC2_Actions.SELECT_ARMY in obs.observation['available_actions'] :
                self.move_number += 1
                sc2Action = actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])
        
        elif self.move_number == 1:
            self.move_number = 0

            do_it = True
            if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == TerranUnit.SCV:
                do_it = False
            
            if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == TerranUnit.SCV:
                do_it = False

            if do_it and SC2_Actions.ATTACK_MINIMAP in obs.observation["available_actions"]:
                # select target
                isInBase = True   
                if self.previous_state[STATE_ENEMY_ARMY_LOCATION_IDX] != STATE_NON_VALID_NUM:
                    isInBase = False
                    target = GetCoord(self.previous_state[STATE_ENEMY_ARMY_LOCATION_IDX], SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX])

                elif self.enemyBaseDestroyed:
                    # find target
                    target = self.FindClosestToEnemyBase(obs)
                    if target[0] < 0:
                        isInBase = True
                    else:
                        isInBase = False
                
                if isInBase:
                    target = self.enemyBaseCoord
                
                self.attackCommands.append(AttackCmd(self.previous_state, target, isInBase))
                sc2Action = actions.FunctionCall(SC2_Actions.ATTACK_MINIMAP, [SC2_Params.NOT_QUEUED, SwapPnt(target)])

        return sc2Action

    def FirstStep(self, obs):
        self.baseTopLeft = True
        self.enemyBaseCoord = SC2_Params.BOTTOMRIGHT_BASE_LOCATION
        self.selfBaseCoord = SC2_Params.TOPLEFT_BASE_LOCATION

        self.move_number = 0

    def LastStep(self, obs):
        # naive attack for now
        a = 9

    def Learn(self, obs):
        # naive attack for now

        # if self.current_action is not None:
        #     self.qTable.learn(str(self.previous_scaled_state), self.current_action, 0, str(self.current_scaled_state))
        #     self.current_action = None

        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def CreateState(self, obs):
        
        self.current_state[STATE_ARMY_POWER_IDX] = obs.observation['player'][SC2_Params.ARMY_SUPPLY]

        enemy_y, enemy_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()

        minDist = SC2_Params.MAX_MINIMAP_DIST

        enemyArmyPower = 0   
        enemyBasePower = 0
        closestEnemyLocationIdx = -1
        for i in range(0, len(enemy_y)):
            y = enemy_y[i]
            x = enemy_x[i]

            if (y > SC2_Params.MINIMAP_SIZE[SC2_Params.Y_IDX] / 2 and x > SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX] / 2):
                enemyBasePower += 1
            else:
                enemyArmyPower += 1 
                y -= self.selfBaseCoord[SC2_Params.Y_IDX]
                x -= self.selfBaseCoord[SC2_Params.X_IDX]
                dist = x * x + y * y
                if (dist < minDist):
                    minDist = dist
                    closestEnemyLocationIdx = i
        

        if enemyArmyPower == 0:
            self.current_state[STATE_ENEMY_ARMY_POWER_IDX] = STATE_NON_VALID_NUM 
            self.current_state[STATE_ENEMY_ARMY_LOCATION_IDX] = STATE_NON_VALID_NUM
        else:
            self.current_state[STATE_ENEMY_ARMY_POWER_IDX] = enemyArmyPower
            self.current_state[STATE_ENEMY_ARMY_LOCATION_IDX] = enemy_x[closestEnemyLocationIdx] + enemy_y[closestEnemyLocationIdx] * SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX]
        
        if enemyBasePower == 0:
            self.current_state[STATE_ENEMY_BASE_POWER_IDX] = STATE_NON_VALID_NUM 
        else:
            self.current_state[STATE_ENEMY_BASE_POWER_IDX] = enemyBasePower            

        self.ScaleCurrState()
   
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

        self.current_scaled_state[STATE_ARMY_POWER_IDX] = math.ceil(self.current_scaled_state[STATE_ARMY_POWER_IDX] / STATE_SELF_ARMY_BUCKETING) * STATE_SELF_ARMY_BUCKETING

        loc = self.current_state[STATE_ENEMY_ARMY_LOCATION_IDX]
        if loc != STATE_NON_VALID_NUM:
            y, x = GetCoord(loc, SC2_Params.MINIMAP_SIZE[SC2_Params.X_IDX])

            self.current_scaled_state[STATE_ENEMY_ARMY_LOCATION_IDX] = ScaleLoc2Grid(x, y)
            self.current_scaled_state[STATE_ENEMY_ARMY_POWER_IDX] = math.ceil(self.current_state[STATE_ENEMY_ARMY_POWER_IDX] / STATE_POWER_BUCKETING) * STATE_POWER_BUCKETING  
        
        if self.current_state[STATE_ENEMY_BASE_POWER_IDX] != STATE_NON_VALID_NUM:
            self.current_scaled_state[STATE_ENEMY_BASE_POWER_IDX] = math.ceil(self.current_state[STATE_ENEMY_BASE_POWER_IDX] / STATE_POWER_BUCKETING) * STATE_POWER_BUCKETING  

    def FindClosestToEnemyBase(self, obs):
        enemyPnt_y, enemyPnt_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()
        if len(enemyPnt_y) > 0:
            return [-1,-1]

        minDist = SC2_Params.MAX_MINIMAP_DIST
        minIdx = -1
        for i in range(0, len(enemyPnt_y)):
            diffX = abs(enemyPnt_x[i] - self.enemyBaseCoord[SC2_Params.X_IDX])
            diffY = abs(enemyPnt_y[i] - self.enemyBaseCoord[SC2_Params.Y_IDX])
            dist = diffX * diffX + diffY * diffY
            if (dist < minDist):
                minDist = dist
                minIdx = i

        return enemyPnt_y[minIdx], enemyPnt_x[minIdx]

    def ComputeAttackResults(self, obs):
        range2Include = 4

        selfMat = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF)
        enemyMat = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE)

        for attack in self.attackCommands[:]:
            if attack.m_inTheWayBattle[SC2_Params.Y_IDX] == -1:
                enemyPower = PowerSurroundPnt(attack.m_attackCoord, range2Include, enemyMat)
                selfPower = PowerSurroundPnt(attack.m_attackCoord, range2Include, selfMat)
                
                if enemyPower > 0:
                    if attack.m_isBaseAttack:
                        powerIdx = STATE_ENEMY_BASE_POWER_IDX
                    else:
                        powerIdx = STATE_ENEMY_ARMY_POWER_IDX

                    attack.m_state[powerIdx] = max(attack.m_state[powerIdx], enemyPower)
                
            else:
                enemyPower = PowerSurroundPnt(attack.m_inTheWayBattle, range2Include, enemyMat)
                selfPower = PowerSurroundPnt(attack.m_inTheWayBattle, range2Include, selfMat)

            if not attack.m_attackStarted:
                isBattle, battleLocation_y, battleLocation_x = BattleStarted(selfMat, enemyMat)
                if isBattle:
                    if abs(attack.m_attackCoord[SC2_Params.Y_IDX] - battleLocation_y) > range2Include or abs(attack.m_attackCoord[SC2_Params.X_IDX] - battleLocation_x) > range2Include:
                        attack.m_inTheWayBattle = [battleLocation_y, battleLocation_x]

                    attack.m_attackStarted = True

            elif not attack.m_attackEnded:
                if enemyPower == 0:    
                    if attack.m_inTheWayBattle[SC2_Params.Y_IDX] == -1:
                        attack.m_attackEnded = True
                    else:
                        attack.m_attackStarted = False
                        attack.m_inTheWayBattle = [-1,-1]

                elif selfPower == 0:
                    attack.m_attackEnded = True
            else:     
                if attack.m_isBaseAttack:
                    self.baseDestroyed = self.IsBaseDestroyed(selfMat, enemyMat, obs)
                # self.tTable.insert(str(attack.m_state), ID_ACTION_ATTACK, str(self.previous_state))
                self.attackCommands.remove(attack)

    def IsBaseDestroyed(self, selfMat, enemyMat, obs, radius2SelfCheck = 6, radius2EnemyCheck = 4):
        # check if self inbase point
        slefInEnemyBase = False
        for x in range (self.enemyBaseCoord[SC2_Params.X_IDX] - radius2SelfCheck, self.enemyBaseCoord[SC2_Params.X_IDX] + radius2SelfCheck):
            for y in range (self.enemyBaseCoord[SC2_Params.Y_IDX] - radius2SelfCheck, self.enemyBaseCoord[SC2_Params.Y_IDX] + radius2SelfCheck):
                if selfMat[y][x]:
                    slefInEnemyBase = True
                    break
            if slefInEnemyBase:
                break

        # if self is not near base return false
        if not slefInEnemyBase:
            return False

        # if enemy is near base than return false
        for x in range (self.enemyBaseCoord[SC2_Params.X_IDX] - radius2EnemyCheck, self.enemyBaseCoord[SC2_Params.X_IDX] + radius2EnemyCheck):
            for y in range (self.enemyBaseCoord[SC2_Params.Y_IDX] - radius2EnemyCheck, self.enemyBaseCoord[SC2_Params.Y_IDX] + radius2EnemyCheck):
                if enemyMat[y][x]:
                    return False
        
        # if self in enemy base and enemy is not near enemy base enemy base is destroyed
        return True
            


