# train army sub agent
import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.lib import actions

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions
from utils_tables import QLearningTable

from utils import SwapPnt
from utils import FindMiddle
from utils import GetScreenCorners
from utils import IsolateArea
from utils import Scale2MiniMap
from utils import GetLocationForBuildingAddition

ID_ACTION_DO_NOTHING = 0
ID_ACTION_BUILD_BARRACKS_ADDITION = 1
ID_ACTION_BUILD_FACTORY_ADDITION = 2
ID_ACTION_TRAIN_BARRACKS = 3
ID_ACTION_TRAIN_FACTORY = 4

NUM_ACTIONS = 5


# state details
STATE_NON_VALID_NUM = -1

STATE_MINERALS_MAX = 500
STATE_GAS_MAX = 300
STATE_MINERALS_BUCKETING = 50
STATE_GAS_BUCKETING = 50

STATE_MINERALS_IDX = 0
STATE_GAS_IDX = 1
STATE_BARRACKS_IDX = 2
STATE_FACTORY_IDX = 3
STATE_BARRACKS_ADDITION_IDX = 3
STATE_FACTORY_ADDITION_IDX = 4
STATE_ARMY_POWER = 5
STATE_QUEUE_BARRACKS = 6
STATE_QUEUE_FACTORY = 7
STATE_SIZE = 8

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

class TrainArmySubAgent:
    def __init__(self, qTableName = ''):        

        # states and action:
        self.current_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        # model params
        self.unit_type = None

        self.cameraCornerNorthWest = [-1,-1]
        self.cameraCornerSouthEast = [-1,-1]

        self.currentBuildingTypeSelected = TerranUnit.BARRACKS
        self.currentBuildingCoordinate = [-1,-1]

        self.IsMultiSelect = False

        self.lastUnitTrain = None

    def step(self, obs):
        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]

        sc2Action = DO_NOTHING_SC2_ACTION

        if self.move_number == 0:
            if self.currentBuildingTypeSelected == TerranUnit.BARRACKS:
                additionType = TerranUnit.REACTOR
            else:
                additionType = TerranUnit.TechLab

            hasAddition = self.SelectRandomSingleBuilding(obs, self.currentBuildingTypeSelected, additionType)
                        
            if random.randint(0, 1) == 0:
                selectAll = hasAddition
            else:
                selectAll = True

            if self.currentBuildingCoordinate[0] >= 0:
                self.move_number += 1
                if selectAll:
                    self.IsMultiSelect = True
                    sc2Action = actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(self.currentBuildingCoordinate)])
                else:
                    self.IsMultiSelect = False
                    sc2Action = actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(self.currentBuildingCoordinate)])
            else:
                self.lastUnitTrain = None
        
        elif self.move_number == 1:
            self.move_number = 0
            self.lastUnitTrain = None
            if self.currentBuildingTypeSelected == TerranUnit.BARRACKS:
                if not self.IsMultiSelect:
                    if SC2_Actions.BUILD_REACTOR in obs.observation['available_actions']:
                        coord = GetLocationForBuildingAddition(obs, TerranUnit.BARRACKS, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                        if coord[0] >= 0:
                            self.nextTrainArmyBuilding = TerranUnit.FACTORY
                            return actions.FunctionCall(SC2_Actions.BUILD_REACTOR, [SC2_Params.QUEUED, SwapPnt(coord)])


                if SC2_Actions.TRAIN_REAPER in obs.observation['available_actions']:
                    self.nextTrainArmyBuilding = TerranUnit.FACTORY
                    self.lastUnitTrain = SC2_Actions.TRAIN_REAPER
                    return actions.FunctionCall(SC2_Actions.TRAIN_REAPER, [SC2_Params.QUEUED])

                if SC2_Actions.TRAIN_MARINE in obs.observation['available_actions']:
                    self.nextTrainArmyBuilding = TerranUnit.FACTORY
                    self.lastUnitTrain = SC2_Actions.TRAIN_MARINE
                    return actions.FunctionCall(SC2_Actions.TRAIN_MARINE, [SC2_Params.QUEUED])

            else:
                if not self.IsMultiSelect:
                    if SC2_Actions.BUILD_TECHLAB in obs.observation['available_actions']:
                        target = GetLocationForBuildingAddition(obs, TerranUnit.FACTORY, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                        if target[0] >= 0:
                            self.lastTrainArmyReward = 0
                            self.nextTrainArmyBuilding = TerranUnit.BARRACKS
                            return actions.FunctionCall(SC2_Actions.BUILD_TECHLAB, [SC2_Params.QUEUED, SwapPnt(target)])
                
                if SC2_Actions.TRAIN_SIEGE_TANK in obs.observation['available_actions']:
                    self.nextTrainArmyBuilding = TerranUnit.BARRACKS
                    self.lastUnitTrain = SC2_Actions.TRAIN_SIEGE_TANK
                    return actions.FunctionCall(SC2_Actions.TRAIN_SIEGE_TANK, [SC2_Params.QUEUED])

                if SC2_Actions.TRAIN_HELLION in obs.observation['available_actions']:
                    self.nextTrainArmyBuilding = TerranUnit.BARRACKS
                    self.lastUnitTrain = SC2_Actions.TRAIN_HELLION
                    return actions.FunctionCall(SC2_Actions.TRAIN_HELLION, [SC2_Params.QUEUED])

        return sc2Action

    def FirstStep(self, obs):
        self.move_number = 0
        self.IsMultiSelect = False
        self.lastUnitTrain = None

    def LastStep(self, obs):
        # naive train army for now
        a = 9

    def Learn(self, obs):
        # naive train army for now

        # if self.current_action is not None:
        #     self.qTable.learn(str(self.previous_scaled_state), self.current_action, 0, str(self.current_scaled_state))
        #     self.current_action = None

        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def CreateState(self, obs):

        self.ScaleCurrState()
   
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]


    def SelectRandomSingleBuilding(self, obs, buildingType, additionType):
        buildingMap = self.unit_type == buildingType
        pnt_y, pnt_x = buildingMap.nonzero()
        if len(pnt_y) > 0:
            
            i = random.randint(0, len(pnt_y) - 1)
            coord = [pnt_y[i], pnt_x[i]]
            onlyBuilding_y, onlyBuilding_x = IsolateArea(coord, buildingMap)
            self.currentBuildingCoordinate = FindMiddle(onlyBuilding_y, onlyBuilding_x)
            return self.HasAddition(onlyBuilding_y, onlyBuilding_x, additionType)
        
        return False

    def HasAddition(self, pnt_y, pnt_x, additionType):
        additionMat = self.unit_type == additionType
        
        for i in range(0, len(pnt_y)):
            nearX = pnt_x[i] + 1
            if nearX < SC2_Params.SCREEN_SIZE and additionMat[pnt_y[i]][nearX]:
                return True

        return False
    def GetLastUnitTrain(self):
        lastUnit = self.lastUnitTrain
        self.lastUnitTrain = None
        return lastUnit
    # def BuildAdditionBarracks(self,obs):
    #     coord = self.SelectBuildingWOAddition(obs, TerranUnit.BARRACKS, TerranUnit.REACTOR)
        
    #     if coord[0] >= 0:



    # def BuildAdditionFactory(self,obs):
    #     coord = self.SelectBuildingWOAddition(obs, TerranUnit.FACTORY, TerranUnit.TECHLAB)

    # def TrainBarracks(self,obs):

    # def TrainFactory(self,obs):

            

    # def SelectBuildingWOAddition(self, obs, buildingType, additionType):
    #     pnt_y, pnt_x = (self.unit_type == TerranUnit.BARRACKS).nonzero()
       
    #     foundAddition = False
    #     while len(pnt_y) > 0:
    #         b_y, b_x = IsolateArea(pnt_y[0], pnt_x[0])
            
    #         if not self.HasAddition(b_y, b_x, additionType):
    #             return FindMiddle(b_y, b_x)
    #         else:
    #             toRemove = []
    #             for matPnt in range(0, len(pnt_y)):
    #                 found = False
    #                 for buildPnt in range(0, len(b_y)):
    #                     if pnt_y[matPnt] == building_y[buildPnt] and pnt_x[matPnt] == building_x[buildPnt]:
    #                         found = True
    #                         break
                    
    #                 if found:
    #                     toRemove.append(matPnt)
                
    #             pnt_y = np.delete(pnt_y, toRemove)
    #             pnt_x = np.delete(pnt_x, toRemove)

    #     return [-1,-1]


