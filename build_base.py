# build base sub agent
import random
import math
import os.path
import time

import numpy as np
import pandas as pd

from pysc2.lib import actions

from train_army import TrainArmySubAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions
from utils_tables import TableMngr
from utils_tables import QLearningTable
from utils_tables import TransitionTable

from utils_tables import QTableParams

from utils import SwapPnt
from utils import FindMiddle
from utils import GetScreenCorners
from utils import IsolateArea
from utils import Scale2MiniMap
from utils import GetLocationForBuilding
from utils import GetLocationForBuildingAddition
from utils import GetLocationForBuildingMiniMap

from utils import PrintScreen
from utils import PrintSpecificMat

ID_ACTION_DO_NOTHING = 0
ID_ACTION_BUILD_SUPPLY_DEPOT = 1
ID_ACTION_BUILD_BARRACKS = 2
ID_ACTION_BUILD_REFINERY = 3
ID_ACTION_BUILD_FACTORY = 4
NUM_ACTIONS = 5

# for solo run
ID_ACTION_TRAIN_ARMY = 5

BUILDING_2_ACTION_TRANSITION = {}
BUILDING_2_ACTION_TRANSITION[TerranUnit.SUPPLY_DEPOT] = ID_ACTION_BUILD_SUPPLY_DEPOT
BUILDING_2_ACTION_TRANSITION[TerranUnit.OIL_REFINERY] = ID_ACTION_BUILD_REFINERY
BUILDING_2_ACTION_TRANSITION[TerranUnit.BARRACKS] = ID_ACTION_BUILD_BARRACKS
BUILDING_2_ACTION_TRANSITION[TerranUnit.FACTORY] = ID_ACTION_BUILD_FACTORY

ACTION_2_BUILDING_TRANSITION = {}
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_SUPPLY_DEPOT] = TerranUnit.SUPPLY_DEPOT
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_REFINERY] = TerranUnit.OIL_REFINERY
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_BARRACKS] = TerranUnit.BARRACKS
ACTION_2_BUILDING_TRANSITION[ID_ACTION_BUILD_FACTORY] = TerranUnit.FACTORY

DO_NOTHING_BUILDING_CHECK = [TerranUnit.COMMANDCENTER, TerranUnit.SUPPLY_DEPOT, TerranUnit.OIL_REFINERY, TerranUnit.BARRACKS, TerranUnit.FACTORY]

# state details
STATE_NON_VALID_NUM = -1

STATE_MINERALS_MAX = 500
STATE_GAS_MAX = 300
STATE_MINERALS_BUCKETING = 50
STATE_GAS_BUCKETING = 50

STATE_COMMAND_CENTER_IDX = 0
STATE_MINERALS_IDX = 1
STATE_GAS_IDX = 2
STATE_SUPPLY_DEPOT_IDX = 3
STATE_REFINERY_IDX = 4
STATE_BARRACKS_IDX = 5
STATE_FACTORY_IDX = 6
STATE_IN_PROGRESS_SUPPLY_DEPOT_IDX = 7
STATE_IN_PROGRESS_REFINERY_IDX = 8
STATE_IN_PROGRESS_BARRACKS_IDX = 9
STATE_IN_PROGRESS_FACTORY_IDX = 10

STATE_SIZE = 11

BUILDING_2_STATE_TRANSITION = {}
BUILDING_2_STATE_TRANSITION[TerranUnit.COMMANDCENTER] = [STATE_COMMAND_CENTER_IDX, -1]
BUILDING_2_STATE_TRANSITION[TerranUnit.SUPPLY_DEPOT] = [STATE_SUPPLY_DEPOT_IDX, STATE_IN_PROGRESS_SUPPLY_DEPOT_IDX]
BUILDING_2_STATE_TRANSITION[TerranUnit.OIL_REFINERY] = [STATE_REFINERY_IDX, STATE_IN_PROGRESS_REFINERY_IDX]
BUILDING_2_STATE_TRANSITION[TerranUnit.BARRACKS] = [STATE_BARRACKS_IDX, STATE_IN_PROGRESS_BARRACKS_IDX]
BUILDING_2_STATE_TRANSITION[TerranUnit.FACTORY] = [STATE_FACTORY_IDX, STATE_IN_PROGRESS_FACTORY_IDX]

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])
DONOTHING_ACTION_NOTHING = 0
DONOTHING_ACTION_LAND_FACTORY = 1
DONOTHING_ACTION_LAND_BARRACKS = 2
DONOTHING_ACTION_IDLE_WORKER = 3
DONOTHING_ACTION_RETURN_SCREEN = 4

class BuildingCoord:
    def __init__(self, screenLocation):
        self.m_screenLocation = screenLocation

class BuildingCmd:
    def __init__(self, screenLocation, inProgress = False):
        self.m_screenLocation = screenLocation
        self.m_inProgress = inProgress
        self.m_steps2Check = 20

class BuildBaseSubAgent:
    def __init__(self, qTableName, tTableName = '', runSolo = False):        
        
        self.runSolo = runSolo
        self.num_Actions = NUM_ACTIONS
        if runSolo:
            self.num_Actions += 1
            self.trainArmySubAgent = TrainArmySubAgent()

        # tables:
        params = QTableParams(STATE_SIZE, self.num_Actions)
        self.tables = TableMngr(params, qTableName)

        if tTableName != '':
            self.use_tTable = True
            self.tTable = TransitionTable(self.num_Actions, tTableName)
        else:
            self.use_tTable = False

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
    
        self.move_number = 0

        self.CommandCenterLoc = []
        self.buildingVec = {}
        self.buildCommands = {}

        self.IsMultiSelect = False
        self.doNothingAction = DONOTHING_ACTION_NOTHING         
        self.Building2Check = 0
        
        # for developing:
        self.bool = False
        self.coord = [-1,-1]

    def step(self, obs, doNothingAction = False):
        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]
               
        if self.move_number == 0:
            if not doNothingAction:
                self.current_action = self.tables.choose_action(str(self.current_scaled_state))
            elif self.current_action != ID_ACTION_DO_NOTHING:
                self.current_action = None

        sc2Action = DO_NOTHING_SC2_ACTION

        if doNothingAction:
            sc2Action = self.DoNothingAction(obs)
        elif self.current_action != ID_ACTION_DO_NOTHING:
            sc2Action = self.BuildAction(obs)

        return sc2Action

    def stepForSolo(self, obs):
        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]
               
        if self.move_number == 0:
            self.current_action = self.tables.choose_action(str(self.current_scaled_state))
            if self.current_action == ID_ACTION_TRAIN_ARMY:
                return "train_army"

        sc2Action = DO_NOTHING_SC2_ACTION

        if self.current_action == ID_ACTION_DO_NOTHING:
            sc2Action = self.DoNothingAction(obs)
        else:
            sc2Action = self.BuildAction(obs)            

        return sc2Action

    def FirstStep(self, obs):
        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]

        player_y, player_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()

        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        self.current_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.move_number = 0
        self.current_state_for_decision_making = None
        
        for key in BUILDING_2_STATE_TRANSITION.keys():
            self.buildingVec[key] = [0]
            self.buildCommands[key] = []

        commandCenterLoc_y, commandCenterLoc_x = (self.unit_type == TerranUnit.COMMANDCENTER).nonzero()

        middleCC = FindMiddle(commandCenterLoc_y, commandCenterLoc_x)
        miniMapLoc = Scale2MiniMap(middleCC, self.cameraCornerNorthWest , self.cameraCornerSouthEast)
        
        self.buildingVec[TerranUnit.COMMANDCENTER][0] += 1
        self.buildingVec[TerranUnit.COMMANDCENTER].append(BuildingCoord(middleCC))
        
        self.Building2Check = 0

        self.CommandCenterLoc = [miniMapLoc]

    def LastStep(self, obs, reward):
        if self.current_action != None:
            self.tables.learn(str(self.previous_scaled_state), self.current_action, reward, 'terminal')
        self.tables.end_run(reward)

    def Learn(self, obs, reward = 0):
        if self.current_action is not None:
            self.tables.learn(str(self.previous_scaled_state), self.current_action, reward, str(self.current_scaled_state))
            self.current_action = None

        self.previous_state[:] = self.current_state[:]
        self.previous_scaled_state[:] = self.current_scaled_state[:]

    def DoNothingAction(self, obs):
        if self.move_number == 0:     
            self.move_number += 1
            # search for flying building
            flyingBa_y, flyingBa_x = (self.unit_type == TerranUnit.FLYING_BARRACKS).nonzero()
            flyingFa_y, flyingFa_x = (self.unit_type == TerranUnit.FLYING_FACTORY).nonzero()
            target = [-1,-1]
            if len(flyingBa_y) > 0:
                i = random.randint(0, len(flyingBa_y) - 1)
                target = [flyingBa_x[i], flyingBa_y[i]]
                self.doNothingAction = DONOTHING_ACTION_LAND_BARRACKS
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, target])

            elif len(flyingFa_y) > 0:
                i = random.randint(0, len(flyingFa_y) - 1)
                target = [flyingFa_x[i], flyingFa_y[i]]
                self.doNothingAction = DONOTHING_ACTION_LAND_FACTORY
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, target])

            elif obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT] > 0:
                self.doNothingAction = DONOTHING_ACTION_IDLE_WORKER
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_ALL])
            
            else:
                cc_y, cc_x = (self.unit_type == TerranUnit.COMMANDCENTER).nonzero()
                ret2Base = False
                if len(cc_y) == 0:
                    ret2Base = True
                else:
                    midPnt = FindMiddle(cc_y, cc_x)
                    maxDiff = max(SC2_Params.SCREEN_SIZE / 2 - midPnt[SC2_Params.X_IDX], SC2_Params.SCREEN_SIZE / 2 - midPnt[SC2_Params.Y_IDX])
                    if abs (maxDiff) > 20:
                        ret2Base = True

                if ret2Base and len(self.CommandCenterLoc) > 0:  
                    self.move_number = 0
                    self.doNothingAction = DONOTHING_ACTION_RETURN_SCREEN
                    coord = self.CommandCenterLoc[0]
                    if SC2_Actions.MOVE_CAMERA in obs.observation['available_actions']:
                        return actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(coord)])

        elif self.move_number == 1:
            self.move_number = 0

            if self.doNothingAction == DONOTHING_ACTION_LAND_BARRACKS:
                target = GetLocationForBuildingAddition(obs, TerranUnit.BARRACKS, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                if target[SC2_Params.Y_IDX] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)])

            if self.doNothingAction == DONOTHING_ACTION_LAND_FACTORY:
                target = GetLocationForBuildingAddition(obs, TerranUnit.FACTORY, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                if target[SC2_Params.Y_IDX] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)])

            if self.doNothingAction == DONOTHING_ACTION_IDLE_WORKER:
                if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                    target = self.GatherHarvest()
                    if target[0] >= 0:
                        return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]) 

        return DO_NOTHING_SC2_ACTION

    def BuildAction(self, obs):
        if self.move_number == 0:
            # select scv
            unit_y, unit_x = (self.unit_type == TerranUnit.SCV).nonzero()
                
            if unit_y.any():
                self.move_number += 1
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, target])
        
        elif self.move_number == 1:
            self.move_number += 1
            buildingType = ACTION_2_BUILDING_TRANSITION[self.current_action]
            sc2Action = TerranUnit.BUIILDING_2_SC2ACTIONS[buildingType]
            if sc2Action in obs.observation['available_actions']:
                coord = GetLocationForBuilding(obs, self.cameraCornerNorthWest, self.cameraCornerSouthEast, buildingType)

                if coord[SC2_Params.Y_IDX] >= 0:
                    self.buildCommands[buildingType].append(BuildingCmd(coord))
                    return actions.FunctionCall(sc2Action, [SC2_Params.NOT_QUEUED, SwapPnt(coord)])

                           
        elif self.move_number == 2:
            self.move_number = 0

            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:

                target = self.GatherHarvest()
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)])
        
        return DO_NOTHING_SC2_ACTION

    def CreateState(self, obs):
        for key, value in BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.buildingVec[key][0]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.buildCommands[key])
     
        self.current_state[STATE_MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[STATE_GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        self.ScaleCurrState()
   
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[STATE_MINERALS_IDX] = math.ceil(self.current_scaled_state[STATE_MINERALS_IDX] / STATE_MINERALS_BUCKETING) * STATE_MINERALS_BUCKETING
        self.current_scaled_state[STATE_MINERALS_IDX] = min(STATE_MINERALS_MAX, self.current_scaled_state[STATE_MINERALS_IDX])
        self.current_scaled_state[STATE_GAS_IDX] = math.ceil(self.current_scaled_state[STATE_GAS_IDX] / STATE_GAS_BUCKETING) * STATE_GAS_BUCKETING
        self.current_scaled_state[STATE_GAS_IDX] = min(STATE_GAS_MAX, self.current_scaled_state[STATE_GAS_IDX])

    def GatherHarvest(self):
        if random.randint(0, 4) < 4:
            resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD
        else:
            resourceList = [TerranUnit.OIL_REFINERY]

        unit_y = []
        unit_x = []
        for val in resourceList[:]:
            p_y, p_x = (self.unit_type == val).nonzero()
            unit_y += list(p_y)
            unit_x += list(p_x)

        if len(unit_y) > 0:
            i = random.randint(0, len(unit_y) - 1)
            return [unit_y[i], unit_x[i]]
        
        return [-1,-1]

    def UpdateBuildingInProgress(self, obs):
        #sort command by in progress
        def getKey(cmd):
            return cmd.m_inProgress

        for building, commands in self.buildCommands.items():
            for cmd in commands[:]:
                if not cmd.m_inProgress:
                    x = cmd.m_screenLocation[SC2_Params.X_IDX]
                    y = cmd.m_screenLocation[SC2_Params.Y_IDX]

                    for off_y in range (-1, 1):
                        for off_x in range (-1, 1):
                            if self.unit_type[y + off_y][x + off_x] == building:
                                cmd.m_inProgress = True
                                break
                        if cmd.m_inProgress:
                            break

                    cmd.m_steps2Check -= 1
                    if cmd.m_steps2Check == 0:
                        self.buildCommands[building].remove(cmd)

            self.buildCommands[building].sort(key = getKey, reverse=True)

    def UpdateBuildingCompletion(self, obs):
        buildingType = DO_NOTHING_BUILDING_CHECK[self.Building2Check]
        self.UpdateSpecificBuildingCompletion(obs, buildingType)

    def UpdateSpecificBuildingCompletion(self, obs, buildingType):  
        buildingStatus = obs.observation['multi_select']
        if len(buildingStatus) == 0:
            buildingStatus = obs.observation['single_select']

        numComplete = 0
        inProgress = 0
        for stat in buildingStatus[:]:
            if stat[SC2_Params.BUILDING_COMPLETION_IDX] == 0:
                numComplete += 1
            else:
                inProgress += 1

        vecInProgress = 0
        for buildingCmd in self.buildCommands[buildingType][:]:
            if buildingCmd.m_inProgress:
                vecInProgress += 1
        
        numBuildingsInVec = self.buildingVec[buildingType][0]

        diff = numComplete - numBuildingsInVec
        # add new buildings
        if diff > 0:
            # specific error reason unknown
            if numBuildingsInVec == 0 and numComplete == 12:
                return
            
            buildingFinished = 0
            for building in self.buildCommands[buildingType][:]:
                if building.m_inProgress:
                    buildingFinished += 1
                    self.buildCommands[buildingType].remove(building)
                
                if buildingFinished == diff:
                    break
            
            numBuildingsInVec += buildingFinished
            diff = numComplete - numBuildingsInVec

        # update num complete to real number
        self.buildingVec[buildingType][0] = numComplete

    def BuidingCheck(self):
        self.Building2Check = (self.Building2Check + 1) % len(DO_NOTHING_BUILDING_CHECK)
        buildingType = DO_NOTHING_BUILDING_CHECK[self.Building2Check]
        self.UpdateScreenBuildings(buildingType)
        numScreenBuilding = len(self.buildingVec[buildingType]) - 1
        if numScreenBuilding > 0:
            idxBuilding = random.randint(1, numScreenBuilding)
            coord = self.buildingVec[buildingType][idxBuilding].m_screenLocation
            return coord
        else:
            return [-1,-1]
        
    def UpdateScreenBuildings(self, buildingType):
        self.buildingVec[buildingType] = [self.buildingVec[buildingType][0]]
        buildingMat = self.unit_type == buildingType
        pnt_y, pnt_x = buildingMat.nonzero()
        if len (pnt_y) > 0:
            while len(pnt_y) > 0:
                building_y, building_x = IsolateArea([pnt_y[0], pnt_x[0]], buildingMat)
                toRemove = []
                for matPnt in range(0, len(pnt_y)):
                    found = False
                    for buildPnt in range(0, len(building_y)):
                        if pnt_y[matPnt] == building_y[buildPnt] and pnt_x[matPnt] == building_x[buildPnt]:
                            found = True
                            break
                    
                    if found:
                        toRemove.append(matPnt)

                pnt_y = np.delete(pnt_y, toRemove)
                pnt_x = np.delete(pnt_x, toRemove)

                coord = FindMiddle(building_y, building_x)
                # wrong assumption : building is built
                self.buildingVec[buildingType].append(BuildingCoord(coord))

