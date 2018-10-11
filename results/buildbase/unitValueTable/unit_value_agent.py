# build base sub agent
import sys
import random
import math
import time
import os.path
import datetime

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils import TableMngr
from utils import QTableParamsWOChangeInExploration
from utils import QTableParamsWithChangeInExploration

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintSpecificMat
from utils import FindMiddle
from utils import DistForCmp

WITH_EXPLORATION_CHANGE = True

UNIT_VALUE_TABLE = "unit_value_table"

NUM_AVAILABLE_UNITS = 6


DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

STEP_DURATION = 0

class ValueTable:
    def __init__(self, values, learningRate = 0.1):

        self.learningRate = learningRate
        self.unitsCol = values

        self.table = pd.DataFrame(columns=self.unitsCol, dtype=np.float)
        if os.path.isfile(UNIT_VALUE_TABLE + '.gz'):
            self.table = pd.read_pickle(UNIT_VALUE_TABLE + '.gz', compression='gzip')
        else:
            for val1 in values[:]:
                self.table = self.table.append(pd.Series([0] * len(self.unitsCol), index = self.unitsCol, name = val1))
                for val2 in values[:]:
                    self.table[val1][val2] = 1
                    self.table[val2][val1] = 1

        print(self.table)
        
    def insert(self, uw, ul, winRatio):

        predictedWinRatio = self.table[uw][ul]
        # if new win ratio is smaller than current(i.e. can win with less soldiers). insert to table
        if winRatio < predictedWinRatio:
            self.table[uw][ul] = predictedWinRatio + self.learningRate * (winRatio - predictedWinRatio)

        lossRatio = 1 / winRatio
        predictedLossRatio = self.table[ul][uw]
        # if new loss ratio is bigger than current(i.e. loss can caused from bigger ratio). insert to table
        if lossRatio > predictedLossRatio:
            self.table[ul][uw] = predictedLossRatio + self.learningRate * (lossRatio - predictedLossRatio)         
 
        self.table.to_pickle(UNIT_VALUE_TABLE+ '.gz', 'gzip') 

class Observe(base_agent.BaseAgent):
    def __init__(self):        
        super(Observe, self).__init__()

        unitNames = []
        for val in TerranUnit.ARMY_SPEC.values():
            unitNames.append(val.name)

        self.valueTable = ValueTable(unitNames)
        self.trialNum = 0 

        self.blueUType = 0
        self.blueUNum = 0
        self.redUType = 0
        self.redUSize = 0

        self.numWins = 0
        self.numLoss = 0
    def step(self, obs):
        super(Observe, self).step(obs)
        if obs.first():
            self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
        

        return DO_NOTHING_SC2_ACTION

    def FirstStep(self, obs):
        self.errorOccur = False

        unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        # unselect army
        self_y, self_x = (obs.observation['feature_screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
        if len(self_y) > 0:
            self.blueUType = unitType[self_y[0]][self_x[0]]
            self.blueUNum = int (obs.observation['player'][SC2_Params.ARMY_SUPPLY] / TerranUnit.ARMY_SPEC[self.blueUType].foodCapacity)
        else:
            self.errorOccur = True

        enemy_y, enemy_x = (obs.observation['feature_screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_HOSTILE).nonzero()
        if len(enemy_y) > 0:
            self.redUType = unitType[enemy_y[0]][enemy_x[0]]
            self.redUNum = int(math.ceil(len(enemy_y) / TerranUnit.ARMY_SPEC[self.redUType].numScreenPixels))
        else:
            self.errorOccur = True

    def ChangeEnemyNum(self, obs):
        a = 9

    def LastStep(self, obs):
        self.trialNum += 1

        if self.errorOccur:
            return
        
        if obs.reward > 0:
            uw = TerranUnit.ARMY_SPEC[self.blueUType].name
            ul = TerranUnit.ARMY_SPEC[self.redUType].name
            ratio = self.blueUNum / self.redUNum
            self.numWins += 1
        elif obs.reward < 0:
            ul = TerranUnit.ARMY_SPEC[self.blueUType].name
            uw = TerranUnit.ARMY_SPEC[self.redUType].name
            ratio = self.redUNum / self.blueUNum
            self.numLoss += 1
        else:
            return

        self.valueTable.insert(uw, ul, ratio)

        if int(self.trialNum) % 20 == 0:
            print(self.valueTable.table)
            print("num wins =", self.numWins, "num loss =", self.numLoss)
    







