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

from agent_resource_mngr import ResourceMngrSubAgent
from agent_resource_mngr import SharedDataResourceMngr

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils_decisionMaker import BaseDecisionMaker
from utils_decisionMaker import BaseNaiveDecisionMaker

from agent_train_army import TrainCmd

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
from utils import GetSelectedUnits

AGENT_NAME = "do_nothing"

SUB_AGENT_RESOURCE_MNGR = 0

ACTION_DO_NOTHING = 0
ACTION_BUILDING_COUNT = 1
ACTION_RESOURCE_SUBAGENT = 2
ACTION_LAND_BUILDING = 3
ACTION_IDLEWORKER_EMPLOYMENT = 4
ACTION_ARMY_COUNT = 5
ACTION_HARVEST_GAS = 6
ACTION_INITIAL_SCV_GROUPING = 7
ACTION_CHECK_QUEUE = 8

NUM_ACTIONS = 9

STATE_INITIAL_GROUPING = 0
STATE_RESOURCE_SUBAGENT_ACTION = 1
STATE_NEW_REFINERY = 2
STATE_FLYING_BUILDING = 3
STATE_NUM_IDLE_SCV = 4
STATE_HAS_RESOURCES = 5
STATE_ARMY_LVL = 6

STATE_SIZE = 7

PRODUCTION_BUILDINGS = [Terran.Barracks, Terran.BarracksReactor, Terran.Factory, Terran.FactoryTechLab]

class SharedDataDoNothing(SharedDataResourceMngr):
    def __init__(self):
        super(SharedDataDoNothing, self).__init__()



class NaiveDecisionMakerDoNothing(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Save = 20):
        super(NaiveDecisionMakerDoNothing, self).__init__(numTrials2Save=numTrials2Save, resultFName=resultFName, directory=directory, agentName=AGENT_NAME)

        self.defaultActionProb = {}
        self.defaultActionProb[ACTION_BUILDING_COUNT] = 0.65
        self.defaultActionProb[ACTION_ARMY_COUNT] = 0.25
        self.defaultActionProb[ACTION_CHECK_QUEUE] = 0.1

        self.counterIdleEmployment = 0
        self.outOfResources = False

    def choose_action(self, state):
        if state[STATE_INITIAL_GROUPING] == 0:
            action = ACTION_INITIAL_SCV_GROUPING
        elif state[STATE_NUM_IDLE_SCV] > 0 and not self.outOfResources: 
            action = ACTION_IDLEWORKER_EMPLOYMENT
        elif state[STATE_RESOURCE_SUBAGENT_ACTION] > 0:
            action = ACTION_RESOURCE_SUBAGENT
        elif state[STATE_NEW_REFINERY] > 0:
            action = ACTION_HARVEST_GAS
        elif state[STATE_FLYING_BUILDING]:
            action = ACTION_LAND_BUILDING
        elif state[STATE_ARMY_LVL] == 0:
            action = ACTION_BUILDING_COUNT
        else: 
            randNum = np.random.uniform()

            for key, prob in self.defaultActionProb.items():
                if randNum < prob:
                    action = key
                    break
                else:
                    randNum -= prob

        if action == ACTION_IDLEWORKER_EMPLOYMENT:
            self.counterIdleEmployment += 1
            if self.counterIdleEmployment == 10:
                self.outOfResources = True
        else:
            self.counterIdleEmployment = 0
        return action

    def ActionValuesVec(self, state, target = True):    
        vals = np.zeros(NUM_ACTIONS, dtype = float)
        vals[self.choose_action(state)] = 1.0

        return vals

    def end_run(self, r, score = 0 ,steps = 0):
        super(NaiveDecisionMakerDoNothing, self).end_run(r, score ,steps)
        self.outOfResources = False



class DoNothingSubAgent(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):
        super(DoNothingSubAgent, self).__init__()

        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList

        self.trainAgent = AGENT_NAME in trainList

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)
        
        self.history = self.decisionMaker.AddHistory()

        # create sub agents and get decision makers
        resMngrDM = self.decisionMaker.GetSubAgentDecisionMaker(SUB_AGENT_RESOURCE_MNGR)  
        self.resourceMngrSubAgent = ResourceMngrSubAgent(sharedData, dmTypes, resMngrDM, isMultiThreaded, saPlayList, trainList)
        self.decisionMaker.SetSubAgentDecisionMaker(SUB_AGENT_RESOURCE_MNGR, self.resourceMngrSubAgent.GetDecisionMaker())

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.isActionFailed = False
        self.current_action = None
     
        self.action2Str = ["DoNothing", "BuildingCount", "ResourceMngr", "LandBuilding", "IdleWorkerEmployment", "ArmyCount", "HarvestGas", "MineralGrouping", "CheckQueue"]

        self.queue2Check = [Terran.Barracks, Terran.Factory] 
        self.addition2QueueCheck = [Terran.BarracksReactor, Terran.FactoryTechLab]      

        self.building2Check = [Terran.CommandCenter, Terran.SupplyDepot, Terran.Refinery, Terran.Barracks, Terran.Factory, Terran.BarracksReactor, Terran.FactoryTechLab] 

        self.sharedData = sharedData

        self.unitInQueue = {}
        self.unitInQueue[Terran.Marine] = [Terran.Barracks, Terran.BarracksReactor]
        self.unitInQueue[Terran.Reaper] = [Terran.Barracks, Terran.BarracksReactor]
        self.unitInQueue[Terran.Hellion] = [Terran.Factory, Terran.FactoryTechLab]
        self.unitInQueue[Terran.SiegeTank] = [Terran.Factory, Terran.FactoryTechLab]

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        if dmTypes[AGENT_NAME] == "none":
            return BaseDecisionMaker(AGENT_NAME)
  
        decisionMaker = NaiveDecisionMakerDoNothing()
        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
            
        return None

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1
        if self.resourceMngrSubAgent.FindActingHeirarchi() >= 0:
            return SUB_AGENT_RESOURCE_MNGR

        return -1
        
    def FirstStep(self, obs):    
        super(DoNothingSubAgent, self).FirstStep()  

        self.resourceMngrSubAgent.FirstStep(obs)

        self.currQueue2Check = 0
        self.idxBuildingQueue2Check = 0
        self.buildingSelected = None

        self.currBuilding2Check = 0
        self.realBuilding2Check = None
        
        self.current_state = np.zeros(STATE_SIZE, int)
        self.previous_state = np.zeros(STATE_SIZE, int)

        self.gasGroup2SentScv = 0
        self.numIdleWorkers = 0

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)
        
        self.resourceMngrSubAgent.EndRun(reward, score, stepNum)

    def MonitorObservation(self, obs):
        self.resourceMngrSubAgent.MonitorObservation(obs)

    def IsDoNothingAction(self, a):
        return True

    def CreateState(self, obs):
        self.resourceMngrSubAgent.CreateState(obs)

        self.numIdleWorkers = obs.observation['player'][SC2_Params.IDLE_WORKER_COUNT]
        unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        
        numCompletedRefineries = len(self.sharedData.buildingCompleted[Terran.Refinery])

        self.resourceMngrSubAgentAction = self.resourceMngrSubAgent.ChooseAction()

        self.current_state[STATE_INITIAL_GROUPING] = self.sharedData.scvMineralGroup != None
        self.current_state[STATE_RESOURCE_SUBAGENT_ACTION] = self.resourceMngrSubAgentAction
        self.current_state[STATE_NEW_REFINERY] = self.gasGroup2SentScv < numCompletedRefineries
        self.current_state[STATE_FLYING_BUILDING] = False
        self.current_state[STATE_NUM_IDLE_SCV] = self.numIdleWorkers
        self.current_state[STATE_HAS_RESOURCES] = self.HasResources(unitType)
        self.current_state[STATE_ARMY_LVL] = self.HasArmyProductionBuildings()

    def SubAgentActionChosen(self, action):
        self.isActionCommitted = True
        self.lastActionCommitted = action

        if action == ACTION_RESOURCE_SUBAGENT:
            self.resourceMngrSubAgent.SubAgentActionChosen(self.resourceMngrSubAgentAction)

    def ChooseAction(self):
        if self.playAgent:
            return self.decisionMaker.choose_action(self.current_state)
        elif self.subAgentPlay == SUB_AGENT_RESOURCE_MNGR:
            return ACTION_RESOURCE_SUBAGENT
        else:
            return ACTION_DO_NOTHING

    def Learn(self, reward, terminal):            
        if self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)

            if self.isActionCommitted:
                self.history.learn(self.previous_state, self.lastActionCommitted, reward, self.current_state, terminal)

        self.resourceMngrSubAgent.Learn(reward, terminal)
        self.previous_state[:] = self.current_state[:]
        self.isActionCommitted = False

    def Action2SC2Action(self, obs, action, moveNum):
        if action == ACTION_BUILDING_COUNT:
            return self.ActionBuildingCount(obs, moveNum)
        elif action == ACTION_RESOURCE_SUBAGENT:
            return self.resourceMngrSubAgent.Action2SC2Action(obs, self.resourceMngrSubAgentAction, moveNum)
        elif action == ACTION_LAND_BUILDING:
            return self.ActionLandBuilding(obs, moveNum)
        elif action == ACTION_IDLEWORKER_EMPLOYMENT:
            return self.ActionIdleWorkerImployment(obs, moveNum)
        elif action == ACTION_ARMY_COUNT:
            return self.ActionArmyCount(obs, moveNum)
        elif action == ACTION_HARVEST_GAS:
            return self.ActionHarvestGas(obs, moveNum)
        elif action == ACTION_INITIAL_SCV_GROUPING:
            return self.CreateMineralsGroup(obs, moveNum)
        elif action == ACTION_CHECK_QUEUE:
            return self.ActionCheckQueue(obs, moveNum)

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
            scvStatus = GetSelectedUnits(obs)
            if len(scvStatus) > 0:
                self.sharedData.scvMineralGroup = self.resourceMngrSubAgent.GetMineralGroup()
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_SET, [self.sharedData.scvMineralGroup]]), False
        elif moveNum == 2:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = GatherResource(unitType, SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionHarvestGas(self, obs, moveNum):
        if moveNum == 0:
            gasGroup = self.sharedData.scvGasGroups[self.gasGroup2SentScv]
            if gasGroup != None:
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, [gasGroup]]), False
        elif moveNum == 1:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                target = self.sharedData.buildingCompleted[Terran.Refinery][self.gasGroup2SentScv].m_screenLocation
                self.gasGroup2SentScv += 1
                return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True

        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

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
            scvStatus = GetSelectedUnits(obs)
            if len(scvStatus) > 0:
                return actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_APPEND_AND_STEAL, [self.sharedData.scvMineralGroup]]), False
        elif moveNum == 2:
            if SC2_Actions.HARVEST_GATHER in obs.observation['available_actions']:
                unitType = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
                target = GatherResource(unitType, SC2_Params.NEUTRAL_MINERAL_FIELD)
                if target[0] < 0:
                    target = GatherResource(unitType, Terran.Refinery)
                
                if target[0] >= 0:
                    return actions.FunctionCall(SC2_Actions.HARVEST_GATHER, [SC2_Params.QUEUED, SwapPnt(target)]), True


        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionLandBuilding(self, obs, moveNum):
        return SC2_Actions.DO_NOTHING_SC2_ACTION, True

    def ActionCheckQueue(self, obs, moveNum):
        if moveNum == 0:
            found = self.FindQueue2Check()
            self.idxBuildingQueue2Check = 0
            if not found:
                return SC2_Actions.DO_NOTHING_SC2_ACTION, True
                   
        if self.idxBuildingQueue2Check > 0:
            self.CorrectQueue(obs.observation['build_queue'])

        buildingType = self.queue2Check[self.currQueue2Check]
        numBuildings = len(self.sharedData.buildingCompleted[buildingType])

        if self.idxBuildingQueue2Check >= numBuildings:
            additionType = self.addition2QueueCheck[self.currQueue2Check]
            idx = self.idxBuildingQueue2Check - numBuildings

            if idx >= len(self.sharedData.buildingCompleted[additionType]):
                self.buildingSelected = None
                self.currQueue2Check = (self.currQueue2Check + 1) % len(self.queue2Check)
                return SC2_Actions.DO_NOTHING_SC2_ACTION, True
            
            self.buildingSelected = self.sharedData.buildingCompleted[additionType][idx]
        else:
            self.buildingSelected = self.sharedData.buildingCompleted[buildingType][self.idxBuildingQueue2Check]

        self.idxBuildingQueue2Check += 1

        target = self.buildingSelected.m_screenLocation
        return actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_SINGLE, SwapPnt(target)]), False
            

    def CorrectQueue(self, buildingQueue):
        stepCompleted = -1000
        self.buildingSelected.qForProduction = []
        for bq in buildingQueue:
            unit = TrainCmd(bq[SC2_Params.UNIT_TYPE_IDX])
            
            if bq[SC2_Params.COMPLETION_RATIO_IDX] > 0:
                unit.step = bq[SC2_Params.COMPLETION_RATIO_IDX]
            else:
                unit.step = stepCompleted
                stepCompleted -= 1

            self.buildingSelected.qForProduction.append(unit)

    def FindQueue2Check(self):
        found = False
        idxBuildingCheck = self.currQueue2Check
        while not found:
            buildingType = self.queue2Check[idxBuildingCheck]
            additionType = self.addition2QueueCheck[idxBuildingCheck]
            if len(self.sharedData.buildingCompleted[buildingType]) >= 0 or len(self.sharedData.buildingCompleted[additionType]) >= 0:
                found = True
                self.currQueue2Check = idxBuildingCheck
            else:
                idxBuildingCheck = (self.currQueue2Check + 1) % len(self.queue2Check)
                if idxBuildingCheck == self.currQueue2Check:
                    return False
        
        return found

    def HasResources(self, unitType):
        resourceList = SC2_Params.NEUTRAL_MINERAL_FIELD + SC2_Params.VESPENE_GAS_FIELD
        for val in resourceList[:]:
            if (unitType == val).any():
                return True
        
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

    def Action2Str(self, a, onlyAgent=False):
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
            if stat[SC2_Params.COMPLETION_RATIO_IDX] == 0:
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


            