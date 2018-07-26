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

from pysc2.agents import base_agent
from pysc2.lib import actions

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

# shared data
from build_base import SharedDataBuild
from build_base import BuildingCmd

from utils import GetScreenCorners
from utils import GetLocationForBuildingAddition
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import SelectBuildingValidPoint

ACTION_BUILDING_COUNT = 0
ACTION_LAND_BARRACKS = 1
ACTION_LAND_FACTORY = 2
ACTION_IDLEWORKER_EMPLOYMENT = 3
ACTION_RETURN2BASE = 4
ACTION_ARMY_COUNT = 5
NUM_ACTIONS = 6

class DoNothingSubAgent:
    def __init__(self):
        self.isActionFailed = False
        self.current_action = None

        self.action2Str = ["BuildingCount", "LandBarracks", "LandFactory", "IdleWorkerEmployment", "Return2Base", "QueueCheck"]
        
        self.unit_type = None
        self.cameraCornerNorthWest = None
        self.cameraCornerSouthEast = None
                  
        self.building2Check = [TerranUnit.COMMANDCENTER, TerranUnit.SUPPLY_DEPOT, TerranUnit.OIL_REFINERY, TerranUnit.BARRACKS, TerranUnit.FACTORY, TerranUnit.REACTOR, TerranUnit.TECHLAB] 

        self.sharedData = None

        self.checkCounter2RemoveCmd = 5

        self.unitInQueue = {}
        self.unitInQueue[TerranUnit.MARINE] = TerranUnit.BARRACKS
        self.unitInQueue[TerranUnit.REAPER] = TerranUnit.BARRACKS
        self.unitInQueue[TerranUnit.HELLION] = TerranUnit.FACTORY
        self.unitInQueue[TerranUnit.SIEGE_TANK] = TerranUnit.TECHLAB


    def step(self, obs, sharedData = None, moveNum = 0):

        self.cameraCornerNorthWest , self.cameraCornerSouthEast = GetScreenCorners(obs)
        self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]
        if obs.first():
            self.FirstStep(obs)
        
        if sharedData != None:
            self.sharedData = sharedData

        if moveNum == 0:
            self.current_action = self.ChooseAction(obs)

        return self.current_action

    def FirstStep(self, obs):
        commandCenterLoc_y, commandCenterLoc_x = (self.unit_type == TerranUnit.COMMANDCENTER).nonzero()
        middleCC = FindMiddle(commandCenterLoc_y, commandCenterLoc_x)
        miniMapLoc = Scale2MiniMap(middleCC, self.cameraCornerNorthWest , self.cameraCornerSouthEast)
        
        self.CommandCenterLoc = [miniMapLoc]
        
        self.currBuilding2Check = 0
        self.sharedData = None
        self.realBuilding2Check = "None"

    def IsDoNothingAction(self, a):
        return True
    
    def Learn(self, reward):
        return

    def ChooseAction(self, obs):
        # if (self.unit_type == TerranUnit.FLYING_BARRACKS).any():
        #     return ACTION_LAND_BARRACKS
        # elif (self.unit_type == TerranUnit.FLYING_FACTORY).any():
        #     return ACTION_LAND_FACTORY
        #el
        if obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT] > 0:
            return ACTION_IDLEWORKER_EMPLOYMENT
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
                return ACTION_RETURN2BASE
            else:
                numBarracks = self.sharedData.buildingCount[TerranUnit.BARRACKS]
                if numBarracks > 0 and np.random.randint(0,4) >= 3:
                    return ACTION_ARMY_COUNT
                else:
                    return ACTION_BUILDING_COUNT

    def Action2SC2Action(self, obs, a, moveNum):
        if moveNum == 0:  
            self.isActionFailed = False 
            finishedAction = False
            if self.current_action == ACTION_BUILDING_COUNT:
                target = self.SelectBuildingPoint()
                if target[0] >= 0 and SC2_Actions.SELECT_POINT in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(target)]), finishedAction
                    

            elif self.current_action == ACTION_ARMY_COUNT:
                if SC2_Actions.SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED]), finishedAction

            elif self.current_action == ACTION_LAND_BARRACKS:
                target = SelectBuildingValidPoint(self.unit_type, TerranUnit.FLYING_BARRACKS)
                if target[0] >= 0:    
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction  

            elif self.current_action == ACTION_LAND_FACTORY :
                target = SelectBuildingValidPoint(self.unit_type, TerranUnit.FLYING_FACTORY)
                if target[0] >= 0:    
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.NOT_QUEUED, SwapPnt(target)]), finishedAction
                          
            elif self.current_action == ACTION_IDLEWORKER_EMPLOYMENT:
                return actions.FunctionCall(SC2_Actions.SELECT_IDLE_WORKER, [SC2_Params.SELECT_ALL]), finishedAction

            elif self.current_action == ACTION_RETURN2BASE:
                coord = self.CommandCenterLoc[0]
                if SC2_Actions.MOVE_CAMERA in obs.observation['available_actions']:
                    return actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(coord)]), finishedAction

        elif moveNum == 1 and not self.isActionFailed:
            finishedAction = True
            if self.current_action == ACTION_BUILDING_COUNT:
                # possible to exploit this move for another check
                self.UpdateBuildingCompletion(obs)
                return SC2_Actions.DO_NOTHING_SC2_ACTION, True

            elif self.current_action == ACTION_ARMY_COUNT:
                # possible to exploit this move for another check
                self.UpdateSoldiersCompletion(obs)
                return SC2_Actions.DO_NOTHING_SC2_ACTION, True


            elif self.current_action == ACTION_LAND_BARRACKS:
                target = GetLocationForBuildingAddition(obs, TerranUnit.BARRACKS, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                if target[SC2_Params.Y_IDX] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), finishedAction

            elif self.current_action == ACTION_LAND_FACTORY :
                target = GetLocationForBuildingAddition(obs, TerranUnit.FACTORY, self.cameraCornerNorthWest, self.cameraCornerSouthEast)
                if target[SC2_Params.Y_IDX] >= 0:
                    return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), finishedAction

            elif self.current_action == ACTION_IDLEWORKER_EMPLOYMENT:
                if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                    target = self.GatherHarvest()
                    if target[0] >= 0:
                        return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), finishedAction
     
        self.isActionFailed = True
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True
    
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

    def LastStep(self, obs, reward):
        return
        
    def SelectBuildingPoint(self):
        prevCheck = self.currBuilding2Check
        wholeRound = False

        while not wholeRound:
            self.currBuilding2Check = (self.currBuilding2Check + 1) % len(self.building2Check)
            
            buildingType = self.building2Check[self.currBuilding2Check] 
            target = SelectBuildingValidPoint(self.unit_type, buildingType)
            if target[0] >= 0:
                self.realBuilding2Check = TerranUnit.UNIT_NAMES[self.building2Check[self.currBuilding2Check]]
                return target

            
            wholeRound = prevCheck == self.currBuilding2Check
        
        self.realBuilding2Check = "None"
        return [-1,-1]

    def Action2Str(self, a):
        if a == ACTION_BUILDING_COUNT and not self.isActionFailed:
            return self.realBuilding2Check + "Count"
        else:
            return self.action2Str[a]

    def UpdateBuildingCompletion(self, obs):
        buildingType = self.building2Check[self.currBuilding2Check]
        buildingStatus = obs.observation['multi_select']
        
        if len(buildingStatus) == 0:
            buildingStatus = obs.observation['single_select']
            if len(buildingStatus) == 0:
                return

        if buildingStatus[0][SC2_Params.UNIT_TYPE_IDX] != buildingType:
            return

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
            if diff > 0:
                print("\n\n\n\nError in calculation of building completion\n\n\n")

        # update num complete to real number
        self.sharedData.buildingCount[buildingType] = numComplete
        self.UpdateBuildingInProgress(inProgress, self.sharedData.buildCommands[buildingType])
    
    def UpdateSoldiersCompletion(self, obs):
        unitType = self.building2Check[self.currBuilding2Check]

        unitStatus = obs.observation['multi_select']
        
        if len(unitStatus) == 0:
            unitStatus = obs.observation['single_select']
            if len(unitStatus) == 0:
                return
        
        
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

                if completedSoldiers != 0:
                    print("\n\n\nError in completed soldiers for unit =", TerranUnit.UNIT_NAMES[key], "\n\n")


            elif count < prevCount:
                print("\n\n\n\t\tdeath!!!\n\n\n")
        
        

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