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

from utils_decisionMaker import BaseDecisionMaker

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

# shared data
from agent_build_base import BuildingCmd

from utils import GetScreenCorners
from utils import GetLocationForBuilding
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import SelectBuildingValidPoint
from utils import SelectUnitValidPoints
from utils import PrintScreen

ACTION_BUILDING_COUNT = 0
ACTION_LAND_BARRACKS = 1
ACTION_LAND_FACTORY = 2
ACTION_IDLEWORKER_EMPLOYMENT = 3
ACTION_RETURN2BASE = 4
ACTION_ARMY_COUNT = 5
ACTION_ATTACK_MONITOR = 6

NUM_ACTIONS = 7

AGENT_NAME = "do_nothing"

CLOSE_TO_ENEMY_RANGE_SQUARE = 10 * 10

class DoNothingSubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):
        super(DoNothingSubAgent, self).__init__()

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)

        self.isActionFailed = False
        self.current_action = None

        self.action2Str = ["BuildingCount", "LandBarracks", "LandFactory", "IdleWorkerEmployment", "Return2Base", "QueueCheck", "Attack_Monitor"]
                       
        self.building2Check = [Terran.CommandCenter, Terran.SupplyDepot, Terran.Refinery, Terran.Barracks, Terran.Factory, Terran.Reactor, Terran.TechLab] 

        self.sharedData = sharedData

        self.checkCounter2RemoveCmd = 5

        self.unitInQueue = {}
        self.unitInQueue[Terran.Marine] = Terran.Barracks
        self.unitInQueue[Terran.Reaper] = Terran.Barracks
        self.unitInQueue[Terran.Hellion] = Terran.Factory
        self.unitInQueue[Terran.SiegeTank] = Terran.TechLab

        self.defaultActionProb = {}
        self.defaultActionProb[ACTION_BUILDING_COUNT] = 0.75
        self.defaultActionProb[ACTION_ARMY_COUNT] = 0.25
        self.defaultActionProb[ACTION_ATTACK_MONITOR] = 0

        
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

    def IsDoNothingAction(self, a):
        return True

    def CreateState(self, obs):
        self.numIdleWorkers = obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT]
        self.unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]


    def ChooseAction(self):
        # if (self.unit_type == Terran.FLYING_BARRACKS).any():
        #     return ACTION_LAND_BARRACKS
        # elif (self.unit_type == Terran.FLYING_FACTORY).any():
        #     return ACTION_LAND_FACTORY
        #el
        if self.numIdleWorkers > 0 and self.HasResources():
            return ACTION_IDLEWORKER_EMPLOYMENT
        else:
            if self.ShouldReturnt2Base(self.unitType) and len(self.sharedData.commandCenterLoc) > 0:  
                return ACTION_RETURN2BASE
            else:
                randNum = np.random.uniform()

                for key, prob in self.defaultActionProb.items():
                    if randNum < prob:
                        action = key
                        break
                    else:
                        randNum -= prob

                if action == ACTION_ATTACK_MONITOR and len(self.sharedData.armyInAttack) > 0:
                    return action
                elif action == ACTION_ARMY_COUNT and self.sharedData.buildingCount[Terran.Barracks] > 0:
                    return action
                else:
                    return ACTION_BUILDING_COUNT

    def Action2SC2Action(self, obs, action, moveNum):
        if moveNum == 0:  
            self.isActionFailed = False 
            finishedAction = False
            if action == ACTION_BUILDING_COUNT:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = self.SelectBuildingPoint(unitType)
                if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), finishedAction
                    

            elif action == ACTION_ARMY_COUNT:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED]), finishedAction

            elif action == ACTION_LAND_BARRACKS:
                target = SelectBuildingValidPoint(self.unitType, Terran.BarracksFlying)
                if target[0] >= 0:    
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction  

            elif action == ACTION_LAND_FACTORY :
                target = SelectBuildingValidPoint(self.unitType, Terran.FactoryFlying)
                if target[0] >= 0:    
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction
                          
            elif action == ACTION_IDLEWORKER_EMPLOYMENT:
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_ALL]), finishedAction

            elif action == ACTION_RETURN2BASE:
                coord = self.sharedData.commandCenterLoc[0]
                if SC2_Actions.MOVE_CAMERA in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(coord)]), finishedAction


        elif moveNum == 1 and not self.isActionFailed:
            finishedAction = True
            cameraCornerNorthWest, cameraCornerSouthEast = GetScreenCorners(obs)
            if action == ACTION_BUILDING_COUNT:
                # possible to exploit this move for another check
                actionSucceed = self.UpdateBuildingCompletion(obs)
                self.isActionFailed = not actionSucceed
                return SC2_Actions.DO_NOTHING_SC2_ACTION, True

            elif action == ACTION_ARMY_COUNT:
                # possible to exploit this move for another check
                actionSucceed = self.UpdateSoldiersCompletion(obs)
                self.isActionFailed = not actionSucceed
                return SC2_Actions.DO_NOTHING_SC2_ACTION, True


            elif action == ACTION_LAND_BARRACKS:
                target = GetLocationForBuilding(obs, Terran.Barracks, Terran.Reactor)
                if target[SC2_Params.Y_IDX] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), finishedAction

            elif action == ACTION_LAND_FACTORY:
                target = GetLocationForBuilding(obs, Terran.Factory, Terran.TechLab)
                if target[SC2_Params.Y_IDX] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), finishedAction

            elif action == ACTION_IDLEWORKER_EMPLOYMENT:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                if self.ShouldReturnt2Base(unitType):
                    # go back to base
                    coord = self.sharedData.commandCenterLoc[0]
                    if SC2_Actions.MOVE_CAMERA in obs.observation['available_actions']:
                        return actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(coord)]), finishedAction

                if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                    target = self.GatherHarvest(unitType)
                    if target[0] >= 0:
                        return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), finishedAction

            elif action == ACTION_ATTACK_MONITOR:
                self.sharedData.armyInAttack = self.UpdateAttackPower(obs)
                self.sharedData.attackStarted = self.SelectedClose2Enemy(obs)

                return SC2_Actions.DO_NOTHING_SC2_ACTION, True

        elif moveNum == 2:
            if action == ACTION_IDLEWORKER_EMPLOYMENT:
                if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                    unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                    target = self.GatherHarvest(unitType)
                    if target[0] >= 0:
                        return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), finishedAction

        self.isActionFailed = True
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True
    
    def ShouldReturnt2Base(self, unitType):
        cc_y, cc_x = (unitType == Terran.CommandCenter).nonzero()
        if len(cc_y) == 0:
            return True  
        
        midPnt = FindMiddle(cc_y, cc_x)
        maxDiff = max(SC2_Params.SCREEN_SIZE / 2 - midPnt[SC2_Params.X_IDX], SC2_Params.SCREEN_SIZE / 2 - midPnt[SC2_Params.Y_IDX])
        return maxDiff > 20

    def GatherHarvest(self, unitType):
        if random.randint(0, 2) < 1:
            resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD
        else:
            resourceList = [Terran.Refinery]

        allResMat = np.in1d(unitType, resourceList).reshape(unitType.shape)
        unit_y, unit_x = SelectUnitValidPoints(allResMat)
        if len(unit_y) > 0:
            i = random.randint(0, len(unit_y) - 1)
            return [unit_y[i], unit_x[i]]
        
        return [-1,-1]

    def HasResources(self):
        resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD + SC2_Params.VESPENE_GAS_FIELD
        for val in resourceList[:]:
            if (self.unitType == val).any():
                return True
        
        #print("\n\n\nout of resources!!\n\n")
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
        

        numBuildingsInVec = self.sharedData.buildingCount[buildingType]

        diff = numComplete - numBuildingsInVec
        # add new buildings
        if diff > 0:         
            buildingFinished = 0
            toRemove = []
            for buildingCmd in self.sharedData.buildCommands[buildingType][:]:
                if buildingCmd.m_inProgress:
                    buildingFinished += 1
                    toRemove.append(buildingCmd)
                
                if buildingFinished == diff:
                    break

            for buildingCmd in toRemove:
                self.sharedData.buildCommands[buildingType].remove(buildingCmd)
            
            numBuildingsInVec += buildingFinished
            diff = numComplete - numBuildingsInVec
            # if diff > 0:
            #     print("\n\n\n\nError in calculation of building completion\n\n\n")

        # update num complete to real number
        self.sharedData.buildingCount[buildingType] = numComplete
        self.UpdateBuildingInProgress(inProgress, self.sharedData.buildCommands[buildingType])

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

        for key in unitCount.keys():
            count = unitCount[key]

            if key in self.sharedData.armySize.keys():
                prevCount = self.sharedData.armySize[key]
                
                self.sharedData.armySize[key] = count
                if count > prevCount:
                    completedSoldiers = count - prevCount
                    qForUnit = self.unitInQueue[key]
                    toRemove = []
                    
                    for u in self.sharedData.trainingQueue[qForUnit]:
                        if u.unitId == key:
                            toRemove.append(u)
                            completedSoldiers -= 1
                        if completedSoldiers == 0:
                            break
                    
                    for rem in toRemove:
                        self.sharedData.trainingQueue[qForUnit].remove(rem)

                # elif count < prevCount:
                #     print("\n\n\n\t\tdeath!!!\n\n\n")

            else:
                return False
        
        return True

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

    def UpdateBuildingInProgress(self, inProgressReal, buildingCmdVec):
        # sort vector according to in progress and counter
        def getKeyCmd(cmd):
            return 100 * cmd.m_inProgress + cmd.m_stepsCounter
        buildingCmdVec = sorted(buildingCmdVec, key=getKeyCmd)

        vecInProgress = 0
        for buildingCmd in buildingCmdVec:
            if vecInProgress < inProgressReal:
                buildingCmd.m_inProgress = True
                vecInProgress += 1
            else:
                buildingCmd.m_inProgress = False

        while vecInProgress < inProgressReal:
            buildingCmdVec.append(BuildingCmd())
            vecInProgress += 1

        toRemoveCmd = []
        for buildingCmd in buildingCmdVec:
            buildingCmd.m_stepsCounter += 1
            if not buildingCmd.m_inProgress and buildingCmd.m_stepsCounter == self.checkCounter2RemoveCmd:
                    toRemoveCmd.append(buildingCmd)

        for buildingCmd in toRemoveCmd:
            buildingCmdVec.remove(buildingCmd)