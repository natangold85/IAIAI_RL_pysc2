import random
import math
import os.path
import sys
import logging
import traceback

#udp
import socket
import threading

import numpy as np
import pandas as pd
import time

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent
from utils import EmptySharedData

from utils_decisionMaker import BaseDecisionMaker

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions


from utils import GetScreenCorners
from utils import GetLocationForBuilding
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import SelectBuildingValidPoint
from utils import SelectUnitValidPoints
from utils import PrintScreen
from utils import GatherResource

ACTION_BUILDING_COUNT = 0
ACTION_LAND_BARRACKS = 1
ACTION_LAND_FACTORY = 2
ACTION_GROUP_SCV = 3
ACTION_IDLEWORKER_EMPLOYMENT = 4
ACTION_ARMY_COUNT = 5
HARVEST_GAS = 6

NUM_ACTIONS = 7

AGENT_NAME = "do_nothing"

CLOSE_TO_ENEMY_RANGE_SQUARE = 10 * 10

PRODUCTION_BUILDINGS = [Terran.Barracks, Terran.BarracksReactor, Terran.Factory, Terran.FactoryTechLab]

SCV_GROUP_MINERALS = 0
SCV_GROUP_GAS1 = 1
SCV_GROUP_GAS2 = 2
GAS_GROUPS = [SCV_GROUP_GAS1, SCV_GROUP_GAS2]
ALL_SCV_GROUPS = [SCV_GROUP_MINERALS] + GAS_GROUPS

class SharedDataResourceMngr(EmptySharedData):
    def __init__(self):
        self.scvBuildingQ = []
        self.scvMineralGroup = SCV_GROUP_MINERALS
        self.scvGasGroup1 = SCV_GROUP_GAS1
        self.scvGasGroup2 = SCV_GROUP_GAS2
        
class ScvCmd:
    def __init__(self):
        self.m_stepsCounter = 0

class DoNothingSubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):
        super(DoNothingSubAgent, self).__init__()

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)

        self.isActionFailed = False
        self.current_action = None

        self.action2Str = ["BuildingCount", "LandBarracks", "LandFactory", "ResourceMngr", "IdleWorkerEmployment", "Return2Base", "QueueCheck", "HarvestGas"]
                       
        self.building2Check = [Terran.CommandCenter, Terran.SupplyDepot, Terran.Refinery, Terran.Barracks, Terran.Factory, Terran.BarracksReactor, Terran.FactoryTechLab] 

        self.sharedData = sharedData

        self.checkCounter2RemoveCmd = 5

        self.unitInQueue = {}
        self.unitInQueue[Terran.Marine] = [Terran.Barracks, Terran.BarracksReactor]
        self.unitInQueue[Terran.Reaper] = [Terran.Barracks, Terran.BarracksReactor]
        self.unitInQueue[Terran.Hellion] = [Terran.Factory, Terran.FactoryTechLab]
        self.unitInQueue[Terran.SiegeTank] = [Terran.Factory, Terran.FactoryTechLab]

        self.defaultActionProb = {}
        self.defaultActionProb[ACTION_BUILDING_COUNT] = 0.75
        self.defaultActionProb[ACTION_ARMY_COUNT] = 0.25

        self.rallyCoordScv = [50,50]
        self.maxQSize = 5

        # required num scv
        self.numScvReq4Group = {}
        self.numScvReq4Group[SCV_GROUP_MINERALS] = 8
        self.numScvReq4Group[SCV_GROUP_GAS1] = 3
        self.numScvReq4Group[SCV_GROUP_GAS2] = 3

        self.numScvRequired = 0
        for req in self.numScvReq4Group.values():
            self.numScvRequired += req

    def GetDecisionMaker(self):
        return None

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        
        return -1
        
    def FirstStep(self, obs):    
        super(DoNothingSubAgent, self).FirstStep()  
        self.currBuilding2Check = 0
        self.realBuilding2Check = None
        
        self.numScvAll = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]

        self.initialGrouping = False
        self.rallyPointSet = False
        self.sharedData.scvBuildingQ = []
        self.scvGroups = {}

        self.sent2HarvestGasLast = 0
        self.prevAction = 0
        self.numIdleWorkers = 0

    def IsDoNothingAction(self, a):
        return True

    def CreateState(self, obs):
        self.numIdleWorkers = obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT]
        self.unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        
        numScvInGroups = 0
        for group in ALL_SCV_GROUPS:
            self.scvGroups[group] = obs.observation['control_groups'][group][SC2_Params.NUM_UNITS_CONTROL_GROUP]
            numScvInGroups += self.scvGroups[group]

        numScv = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION]        
        for i in range(self.numScvAll, numScv):
            self.sharedData.scvBuildingQ.pop(0)
        
        self.numScvAll = numScv

        self.numCompletedRefineries = len(self.sharedData.buildingCompleted[Terran.Refinery])

    def ChooseAction(self):
        # if (self.unit_type == Terran.FLYING_BARRACKS).any():
        #     return ACTION_LAND_BARRACKS
        # elif (self.unit_type == Terran.FLYING_FACTORY).any():
        #     return ACTION_LAND_FACTORY
        if not self.initialGrouping:
            action = ACTION_GROUP_SCV
        elif self.ScvGroupsNotBalanced():
            action = ACTION_GROUP_SCV
        elif self.Need2HarvestGas():
            action = HARVEST_GAS
        elif self.numIdleWorkers > 0 and self.HasResources():
            action = ACTION_IDLEWORKER_EMPLOYMENT
        else:
            randNum = np.random.uniform()

            for key, prob in self.defaultActionProb.items():
                if randNum < prob:
                    action = key
                    break
                else:
                    randNum -= prob

            if action == ACTION_ARMY_COUNT and self.HasArmyProductionBuildings() > 0:
                action = action
            else:
                action = ACTION_BUILDING_COUNT

        return action

    def Action2SC2Action(self, obs, action, moveNum):
        self.prevAction = action
        if action == ACTION_BUILDING_COUNT:
            return self.ActionBuildingCount(obs, moveNum)
        elif action == ACTION_ARMY_COUNT:
            return self.ActionArmyCount(obs, moveNum)
        elif action == ACTION_GROUP_SCV:
            return self.ActionGrouping(obs, moveNum)
        elif action == ACTION_IDLEWORKER_EMPLOYMENT:
            return self.ActionIdleWorkerImployment(obs, moveNum)
        elif action == ACTION_BUILDING_COUNT:
            return self.ActionBuildingCount(obs, moveNum)
        elif action == HARVEST_GAS:
            return self.ActionHarvestGas(obs, moveNum)
    
    def ScvGroupsNotBalanced(self):
        for key in self.numScvReq4Group.keys():
            if self.scvGroups[key] > self.numScvReq4Group[key]:
                return True

        return False

    def ActionGrouping(self, obs, moveNum):
        if self.scvGroups[SCV_GROUP_MINERALS] < self.numScvReq4Group[SCV_GROUP_MINERALS]:
            return self.CreateMineralsGroup(obs, moveNum)
        elif self.ShouldCreateScv(obs):
            return self.CreateScv(obs, moveNum)
        elif self.scvGroups[SCV_GROUP_GAS1] < self.numScvReq4Group[SCV_GROUP_GAS1] and self.scvGroups[SCV_GROUP_MINERALS] > self.numScvReq4Group[SCV_GROUP_MINERALS]:
            return self.Add2GasGroup(obs, moveNum, 0)
        elif self.scvGroups[SCV_GROUP_GAS2] < self.numScvReq4Group[SCV_GROUP_GAS2] and self.scvGroups[SCV_GROUP_MINERALS] > self.numScvReq4Group[SCV_GROUP_MINERALS]:
            return self.Add2GasGroup(obs, moveNum, 1)

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def CreateMineralsGroup(self, obs, moveNum):
        if moveNum == 0:
            # select scv
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            unit_y, unit_x = SelectUnitValidPoints(unitType == Terran.SCV)                    
            if len(unit_y) > 0:
                target = [unit_x[0], unit_y[0]]
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, target]), False
        if moveNum == 1:
            scvStatus = self.GetSelectedUnits(obs)
            if len(scvStatus) > 0:
                self.initialGrouping = True
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_SET, [SCV_GROUP_MINERALS]]), False
        elif moveNum == 2:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = GatherResource(unitType, SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def Add2GasGroup(self, obs, moveNum, gasGroup):
        if moveNum == 0:
            return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, [SCV_GROUP_MINERALS]]), False

        elif moveNum == 1:
            unitSelected = obs.observation['feature_screen'][SC2_Params.SELECTED_IN_SCREEN]
            unit_y, unit_x = SelectUnitValidPoints(unitSelected != 0) 
            if len(unit_y) > 0:
                target = [unit_x[0], unit_y[0]]
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, target]), False

        elif moveNum == 2:
            scvStatus = self.GetSelectedUnits(obs)
            terminal = False if gasGroup < self.numCompletedRefineries else True
            if len(scvStatus) == 1:
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_APPEND_AND_STEAL, [GAS_GROUPS[gasGroup]]]), terminal

        elif moveNum == 3:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.sharedData.buildingCompleted[Terran.Refinery][gasGroup].m_screenLocation
                return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ShouldCreateScv(self, obs):
        need2Create = obs.observation['player'][SC2_Params.WORKERS_SUPPLY_OCCUPATION] + len(self.sharedData.scvBuildingQ) < self.numScvRequired
        qNotLimit = len(self.sharedData.scvBuildingQ) < self.maxQSize
        return need2Create & qNotLimit

    def CreateScv(self, obs, moveNum):
        if moveNum == 0:
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            target = SelectBuildingValidPoint(unitType, Terran.CommandCenter)
            if target[0] >= 0:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), False
        
        elif moveNum == 1:
            if SC2_Actions.TRAIN_SCV in obs.observation['available_actions']:
                terminal = self.rallyPointSet
                self.sharedData.scvBuildingQ.append(ScvCmd())
                return actions.FunctionCall(SC2_Actions.TRAIN_SCV, [SC2_Params.QUEUED]), terminal

        if moveNum == 2:
            if SC2_Actions.RALLY_SCV in obs.observation['available_actions']:
                coord = self.rallyCoordScv
                self.rallyPointSet = True
                return actions.FunctionCall(SC2_Actions.RALLY_SCV, [SC2_Params.NOT_QUEUED, coord]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True 
    
    def Need2HarvestGas(self):
        return self.sent2HarvestGasLast < len(self.sharedData.buildingCompleted[Terran.Refinery])

    def ActionHarvestGas(self, obs, moveNum):
        if moveNum == 0:
            gasGroup = GAS_GROUPS[self.sent2HarvestGasLast]
            return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, [gasGroup]]), False
        elif moveNum == 1:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.sharedData.buildingCompleted[Terran.Refinery][self.sent2HarvestGasLast].m_screenLocation
                self.sent2HarvestGasLast += 1
                return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True

    def ActionBuildingCount(self, obs, moveNum):
        if moveNum == 0:  
            unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
            target = self.SelectBuildingPoint(unitType)
            if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), False

        elif moveNum == 1 and not self.isActionFailed:
            # possible to exploit this move for another check
            self.UpdateBuildingCompletion(obs)
            return SC2_Actions.DO_NOTHING_SC2_ACTION, True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionArmyCount(self, obs, moveNum):
        if moveNum == 0:  
            self.isActionFailed = False 
            if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED]), False
        
        elif moveNum == 1 and not self.isActionFailed:
            # possible to exploit this move for another check
            actionSucceed = self.UpdateSoldiersCompletion(obs)
            self.isActionFailed = not actionSucceed
            return SC2_Actions.DO_NOTHING_SC2_ACTION, True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionIdleWorkerImployment(self, obs, moveNum):
        if moveNum == 0:
            if SC2_Actions.SELECT_IDLE_WORKER in obs.observation['available_actions']:
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_ALL]), False
        elif moveNum == 1:
            scvStatus = self.GetSelectedUnits(obs)
            if len(scvStatus) > 0:
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_APPEND_AND_STEAL, [SCV_GROUP_MINERALS]]), False
        elif moveNum == 2:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = GatherResource(unitType, SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] < 0:
                    target = GatherResource(unitType, Terran.Refinery)
                
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True


        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionAttackMonitor(self, obs, moveNum):
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True


    def ShouldReturnt2Base(self, unitType):
        cc_y, cc_x = (unitType == Terran.CommandCenter).nonzero()
        if len(cc_y) == 0:
            return True  
        
        midPnt = FindMiddle(cc_y, cc_x)
        maxDiff = max(SC2_Params.SCREEN_SIZE / 2 - midPnt[SC2_Params.X_IDX], SC2_Params.SCREEN_SIZE / 2 - midPnt[SC2_Params.Y_IDX])
        return maxDiff > 20

    def GetSelectedUnits(self, obs):
        scvStatus = list(obs.observation['multi_select'])
        if len(scvStatus) ==  0:
            scvStatus = list(obs.observation['single_select'])
        return scvStatus

    def HasResources(self):
        resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD + SC2_Params.VESPENE_GAS_FIELD
        for val in resourceList[:]:
            if (self.unitType == val).any():
                return True
        
        #print("\n\n\nout of resources!!\n\n")
        return False

    def HasArmyProductionBuildings(self):
        for productionBuilding in PRODUCTION_BUILDINGS:
            if len(self.sharedData.buildingCompleted[productionBuilding]) > 0:
                return True
        
        return False

    def SelectBuildingPoint(self, unitType):
        prevCheck = self.currBuilding2Check
        wholeRound = False

        while not wholeRound:
            self.currBuilding2Check = (self.currBuilding2Check + 1) % len(self.building2Check)
            
            buildingType = self.building2Check[self.currBuilding2Check] 
            target = SelectBuildingValidPoint(unitType, buildingType)
            if target[0] >= 0:
                self.realBuilding2Check = buildingType
                return target

            
            wholeRound = prevCheck == self.currBuilding2Check
        
        self.realBuilding2Check = None
        return [-1,-1]

    def Action2Str(self, a):
        if a == ACTION_BUILDING_COUNT and not self.isActionFailed and self.realBuilding2Check != None:
            return TerranUnit.BUILDING_SPEC[self.realBuilding2Check].name + "Count"
        else:
            return self.action2Str[a]

    def UpdateBuildingCompletion(self, obs):
        buildingType = self.building2Check[self.currBuilding2Check]
        buildingStatus = obs.observation['multi_select']
        
        if len(buildingStatus) == 0:
            buildingStatus = obs.observation['single_select']
            if len(buildingStatus) == 0:
                return False

        if buildingStatus[0][SC2_Params.UNIT_TYPE_IDX] != buildingType:
            return False

        numComplete = 0
        inProgress = 0
        for stat in buildingStatus[:]:
            if stat[SC2_Params.BUILDING_COMPLETION_IDX] == 0:
                numComplete += 1
            else:
                inProgress += 1
        

        numBuildingsCompleted = len(self.sharedData.buildingCompleted[buildingType])

        diff = numComplete - numBuildingsCompleted
        # add new buildings
        if diff > 0:         
            # sort commands according time in built
            self.sharedData.buildCommands[buildingType].sort(key=lambda cmd : cmd.m_inProgress * cmd.m_stepsCounter, reverse=True)
            
            completedCmd = []      
            for buildingCmd in self.sharedData.buildCommands[buildingType][:]:
                if not buildingCmd.m_inProgress or diff == 0:
                    break
                
                diff -= 1
                completedCmd.append(buildingCmd)

            for buildingCmd in completedCmd:
                self.sharedData.buildingCompleted[buildingType].append(buildingCmd)
                self.sharedData.buildCommands[buildingType].remove(buildingCmd)
            
            # if diff > 0:
            #     print("\n\n\n\nError in calculation of building completion\n\n\n")

        return True
    
    def UpdateSoldiersCompletion(self, obs):
        unitStatus = obs.observation['multi_select']
        
        if len(unitStatus) == 0:
            unitStatus = obs.observation['single_select']
            if len(unitStatus) == 0:
                return False
        
        
        # count army
        unitCount = {}
        for unit in unitStatus:
            uType = unit[SC2_Params.UNIT_TYPE_IDX]
            
            if uType in unitCount:
                unitCount[uType] += 1
            else:
                unitCount[uType] = 1

        for unit in unitCount.keys():
            count = unitCount[unit]
            currCount = self.sharedData.armySize[unit]

            queuesTypes = self.unitInQueue[unit]
            while count > currCount:
                self.UnitCompleted(unit, queuesTypes)
                currCount += 1
                
            self.sharedData.armySize[unit] = count

        return True

    def UnitCompleted(self, unit, queuesTypes):
        minFirstInQ4Building = -10000            
        buildingCompleted = None

        for buildingType in queuesTypes:
            for building in self.sharedData.buildingCompleted[buildingType]:
                firstInQ = self.UnitInProductionSteps(building.qForProduction, unit)
                
                if firstInQ > minFirstInQ4Building:
                    minFirstInQ4Building = firstInQ
                    buildingCompleted = building
        
        
        if buildingCompleted != None:
            buildingCompleted.RemoveUnitFromQ(unit)

    
    def UnitInProductionSteps(self, q, unit):
        for u in q:
            if u.unit == unit:
                return u.step
        
        return -10000

    def UpdateAttackPower(self, obs):
        unitCount = {}
        unitStatus = obs.observation['multi_select']
        
        if len(unitStatus) == 0:
            unitStatus = obs.observation['single_select']
            if len(unitStatus) == 0:
                return unitCount
        
        
        # count army
        for unit in unitStatus:
            uType = unit[SC2_Params.UNIT_TYPE_IDX]
            if uType in unitCount:
                unitCount[uType] += 1
            else:
                unitCount[uType] = 1

        return unitCount
        
    def SelectedClose2Enemy(self, obs):
        s_y, s_x = obs.observation['feature_minimap'][SC2_Params.SELECTED_IN_MINIMAP].nonzero()
        e_y, e_x = (obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] == SC2_Params.PLAYER_HOSTILE).nonzero()

        for s in range(len(s_y)):
            for e in range(len(e_y)):
                diffY = s_y[s] - e_y[e]
                diffX = s_x[s] - e_x[e]
                dist = diffY * diffY + diffX * diffX
                if dist < CLOSE_TO_ENEMY_RANGE_SQUARE:
                    #self.PrintSel(obs)
                    return True
        
        return False

    def PrintSel(self, obs):
        selectedMat = obs.observation['feature_minimap'][SC2_Params.SELECTED_IN_MINIMAP]
        enemyMat = (obs.observation['feature_minimap'][SC2_Params.PLAYER_RELATIVE_MINIMAP] == SC2_Params.PLAYER_HOSTILE)
        print("\n\nmonitor attack:\n")
        for y in range(64):
            for x in range(64):
                if selectedMat[y][x] != 0:
                    print("s", end = ' ')
                elif enemyMat[y][x] != 0:
                    print("e", end = ' ')
                else:
                    print("_", end = ' ')
            
            print("|")