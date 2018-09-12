import random
import math
import os.path
import sys
import logging
import traceback
import datetime
from multiprocessing import Lock

import numpy as np
import pandas as pd
import time

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_decisionMaker import LearnWithReplayMngr
from utils_decisionMaker import UserPlay
from utils_decisionMaker import BaseNaiveDecisionMaker

from utils_results import PlotResults
from utils_results import ResultFile


from utils_qtable import QTableParamsExplorationDecay
from utils_dqn import DQN_PARAMS

from agent_build_base import BuildBaseSubAgent
from agent_train_army import TrainArmySubAgent
from agent_do_nothing import DoNothingSubAgent

# shared data
from agent_build_base import SharedDataBuild
from agent_train_army import SharedDataTrain
# from agent_resource_mngr import SharedDataGather

# state data
from agent_build_base import BUILD_STATE
from agent_train_army import TRAIN_STATE


from utils import GetScreenCorners
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import IsolateArea

AGENT_DIR = "BaseMngr/"

AGENT_NAME = "base_mngr"
TRAIN_SUB_AGENTS = "base_mngr_subAgents"

ACTION_DO_NOTHING = 0
ACTION_BUILD_BASE = 1
ACTION_TRAIN_ARMY = 2
NUM_ACTIONS = 3

SUB_AGENT_BUILDER = 1
SUB_AGENT_TRAINER = 2
ALL_SUB_AGENTS = [SUB_AGENT_BUILDER, SUB_AGENT_TRAINER]

ACTION2STR = ["DoNothing", "BuildBase", "TrainArmy"]

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_BUILDER] = "BuildBaseSubAgent"
SUBAGENTS_NAMES[SUB_AGENT_TRAINER] = "TrainArmySubAgent"

class SharedDataBase(SharedDataBuild, SharedDataTrain):
    def __init__(self):
        super(SharedDataBase, self).__init__()
        self.currBaseState = None

class BASE_STATE:
    # state details
    NON_VALID_NUM = -1

    MINERALS_MAX = 500
    GAS_MAX = 300
    MINERALS_BUCKETING = 50
    GAS_BUCKETING = 50

    SUPPLY_LEFT_MAX = 10
    SUPPLY_LEFT_BUCKETING = 2

    TIME_LINE_BUCKETING = 1500

    COMMAND_CENTER_IDX = 0
    MINERALS_IDX = 1
    GAS_IDX = 2
    SUPPLY_DEPOT_IDX = 3
    REFINERY_IDX = 4
    BARRACKS_IDX = 5
    FACTORY_IDX = 6
    REACTORS_IDX = 7
    TECHLAB_IDX = 8

    IN_PROGRESS_SUPPLY_DEPOT_IDX = 9
    IN_PROGRESS_REFINERY_IDX = 10
    IN_PROGRESS_BARRACKS_IDX = 11
    IN_PROGRESS_FACTORY_IDX = 12
    IN_PROGRESS_REACTORS_IDX = 13
    IN_PROGRESS_TECHLAB_IDX = 14    

    SUPPLY_LEFT = 15

    QUEUE_BARRACKS = 16
    QUEUE_FACTORY = 17
    QUEUE_TECHLAB = 18

    ARMY_POWER = 19
    TIME_LINE = 20
    SIZE = 21

    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS = {}
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[COMMAND_CENTER_IDX] = BUILD_STATE.COMMAND_CENTER_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[SUPPLY_DEPOT_IDX] = BUILD_STATE.SUPPLY_DEPOT_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[REFINERY_IDX] = BUILD_STATE.REFINERY_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[BARRACKS_IDX] = BUILD_STATE.BARRACKS_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[FACTORY_IDX] = BUILD_STATE.FACTORY_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[REACTORS_IDX] = BUILD_STATE.REACTORS_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[TECHLAB_IDX] = BUILD_STATE.TECHLAB_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[IN_PROGRESS_SUPPLY_DEPOT_IDX] = BUILD_STATE.IN_PROGRESS_SUPPLY_DEPOT_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[IN_PROGRESS_REFINERY_IDX] = BUILD_STATE.IN_PROGRESS_REFINERY_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[IN_PROGRESS_BARRACKS_IDX] = BUILD_STATE.IN_PROGRESS_BARRACKS_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[IN_PROGRESS_FACTORY_IDX] = BUILD_STATE.IN_PROGRESS_FACTORY_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[IN_PROGRESS_REACTORS_IDX] = BUILD_STATE.IN_PROGRESS_REACTORS_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[IN_PROGRESS_TECHLAB_IDX] = BUILD_STATE.IN_PROGRESS_TECHLAB_IDX
    STATE_IDX_MNGR_2_BUILDER_TRANSITIONS[SUPPLY_LEFT] = BUILD_STATE.SUPPLY_LEFT_IDX


    STATE_IDX_MNGR_2_TRAINER_TRANSITIONS = {}
    STATE_IDX_MNGR_2_TRAINER_TRANSITIONS[QUEUE_BARRACKS] = TRAIN_STATE.QUEUE_BARRACKS
    STATE_IDX_MNGR_2_TRAINER_TRANSITIONS[QUEUE_FACTORY] = TRAIN_STATE.QUEUE_FACTORY
    STATE_IDX_MNGR_2_TRAINER_TRANSITIONS[QUEUE_TECHLAB] = TRAIN_STATE.QUEUE_FACTORY_WITH_TECHLAB
    STATE_IDX_MNGR_2_TRAINER_TRANSITIONS[ARMY_POWER] = TRAIN_STATE.ARMY_POWER

    
    TRAIN_BUILDING_RELATED_IDX = [BARRACKS_IDX, FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION = {}
    BUILDING_2_STATE_TRANSITION[Terran.CommandCenter] = [COMMAND_CENTER_IDX, -1]
    BUILDING_2_STATE_TRANSITION[Terran.SupplyDepot] = [SUPPLY_DEPOT_IDX, IN_PROGRESS_SUPPLY_DEPOT_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Refinery] = [REFINERY_IDX, IN_PROGRESS_REFINERY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Barracks] = [BARRACKS_IDX, IN_PROGRESS_BARRACKS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Factory] = [FACTORY_IDX, IN_PROGRESS_FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.BarracksReactor] = [REACTORS_IDX, IN_PROGRESS_REACTORS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.FactoryTechLab] = [TECHLAB_IDX, IN_PROGRESS_TECHLAB_IDX]

    BUILDING_2_STATE_QUEUE_TRANSITION = {}

    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Barracks] = QUEUE_BARRACKS
    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Factory] = QUEUE_FACTORY
    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.FactoryTechLab] = QUEUE_TECHLAB

    IDX2STR = {}

    IDX2STR[COMMAND_CENTER_IDX] = "CC"
    IDX2STR[MINERALS_IDX] = "MIN"
    IDX2STR[GAS_IDX] = "GAS"
    IDX2STR[SUPPLY_DEPOT_IDX] = "SD"
    IDX2STR[REFINERY_IDX] = "REF"
    IDX2STR[BARRACKS_IDX] = "BA"
    IDX2STR[FACTORY_IDX] = "FA"
    IDX2STR[REACTORS_IDX] =  "REA"
    IDX2STR[TECHLAB_IDX] = "TECH"

    IDX2STR[IN_PROGRESS_SUPPLY_DEPOT_IDX] = "SD_B"
    IDX2STR[IN_PROGRESS_REFINERY_IDX] = "REF_B"
    IDX2STR[IN_PROGRESS_BARRACKS_IDX] = "BA_B"
    IDX2STR[IN_PROGRESS_FACTORY_IDX] = "FA_B"
    IDX2STR[IN_PROGRESS_REACTORS_IDX] = "REA_B"
    IDX2STR[IN_PROGRESS_TECHLAB_IDX] = "TECH_B"

    IDX2STR[QUEUE_BARRACKS] = "BA_Q"
    IDX2STR[QUEUE_FACTORY] = "FA_Q"
    IDX2STR[QUEUE_TECHLAB] = "TE_Q"

    IDX2STR[SUPPLY_LEFT] = "Supply_left"
    IDX2STR[ARMY_POWER] = "ArmyPower"

# possible types of play

QTABLE = 'q'
DQN = 'dqn'
USER_PLAY = 'play'
NAIVE = 'naive'

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "hist"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

ALL_TYPES = set([USER_PLAY, QTABLE, DQN, NAIVE])

# table names
RUN_TYPES = {}

RUN_TYPES[QTABLE] = {}
RUN_TYPES[QTABLE][TYPE] = "QLearningTable"
RUN_TYPES[QTABLE][PARAMS] = QTableParamsExplorationDecay(BASE_STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[QTABLE][DECISION_MAKER_NAME] = "baseMngr_qtable"
RUN_TYPES[QTABLE][DIRECTORY] = "baseMngr_qtable"
RUN_TYPES[QTABLE][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[QTABLE][RESULTS] = "baseMngr_result"

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(BASE_STATE.SIZE, NUM_ACTIONS)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "baseMngr_dqn"
RUN_TYPES[DQN][DIRECTORY] = "baseMngr_dqn"
RUN_TYPES[DQN][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[DQN][RESULTS] = "baseMngr_result"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "baseMngr_naive"
RUN_TYPES[NAIVE][RESULTS] = "baseMngr_result"


class NaiveDecisionMakerBaseMngr(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Learn = 20):
        super(NaiveDecisionMakerBaseMngr, self).__init__(numTrials2Learn, resultFName, directory)

    def choose_action(self, state):
        action = 0

        numSDAll = state[BASE_STATE.SUPPLY_DEPOT_IDX] + state[BASE_STATE.IN_PROGRESS_SUPPLY_DEPOT_IDX]
        numRefAll = state[BASE_STATE.REFINERY_IDX] + state[BASE_STATE.IN_PROGRESS_REFINERY_IDX]
        numBaAll = state[BASE_STATE.BARRACKS_IDX] + state[BASE_STATE.IN_PROGRESS_BARRACKS_IDX]
        numFaAll = state[BASE_STATE.FACTORY_IDX] + state[BASE_STATE.IN_PROGRESS_FACTORY_IDX]
        numReactorsAll = state[BASE_STATE.REACTORS_IDX] + state[BASE_STATE.IN_PROGRESS_REACTORS_IDX]
        numTechAll = state[BASE_STATE.TECHLAB_IDX] + state[BASE_STATE.IN_PROGRESS_TECHLAB_IDX]
        numBarracksQ = state[BASE_STATE.QUEUE_BARRACKS]
        
        supplyLeft = state[BASE_STATE.SUPPLY_LEFT]
        power = state[BASE_STATE.ARMY_POWER]

        if supplyLeft <= 2:
            action = ACTION_BUILD_BASE
        elif numSDAll < 3 or numRefAll < 2 or numBaAll < 1 or numReactorsAll < 1:
            action = ACTION_BUILD_BASE
        elif numBarracksQ < 6 and power < 8:
            action = ACTION_TRAIN_ARMY
        elif numFaAll < 1 and numTechAll < 1: 
            action = ACTION_BUILD_BASE
        elif supplyLeft > 5:
            action = ACTION_TRAIN_ARMY
        elif numReactorsAll < 2:
            action = ACTION_BUILD_BASE
        elif supplyLeft > 5:
            action = ACTION_TRAIN_ARMY
        elif supplyLeft < 5:
            action = ACTION_BUILD_BASE
        else:
            action = ACTION_TRAIN_ARMY

        return action

    def ActionValuesVec(self, state, target = True):    
        vals = np.zeros(NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state)] = 1.0

        return vals

class BaseMngr(BaseAgent):
    def __init__(self, sharedData, dmTypes, decisionMaker, isMultiThreaded, playList, trainList):
        super(BaseMngr, self).__init__(BASE_STATE.SIZE)
        
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList
        
        self.trainAgent = AGENT_NAME in trainList
        self.trainSubAgentsSimultaneously = TRAIN_SUB_AGENTS in trainList
  
        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        # create sub agents and get decision makers
        self.subAgents = {}

        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)     
            self.subAgents[key] = saClass(sharedData, dmTypes, saDM, isMultiThreaded, saPlayList, trainList)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())


        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        if self.trainSubAgentsSimultaneously:
            self.SetSubAgentTrainSwitch(dmTypes["numTrial2Train"], [SUB_AGENT_BUILDER, SUB_AGENT_TRAINER])

        self.sharedData = sharedData
        self.terminalState = np.zeros(BASE_STATE.SIZE, dtype=np.int, order='C')


        # model params 
        self.minPriceMinerals = 50

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        
        runType = RUN_TYPES[dmTypes[AGENT_NAME]]
        directory = dmTypes["directory"] + "/" + AGENT_DIR

        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerBaseMngr(resultFName=runType[RESULTS], directory=directory + runType[DIRECTORY])
        else:        
            decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                                resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory + runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def SetSubAgentTrainSwitch(self, numTrial2Switch, trainOrder):
        self.numTrial2Switch = numTrial2Switch
        self.trainOrder = trainOrder
        self.switchCounter = {}
        
        startAgent = 0
        minCounter = 1000
        for idx in range(len(trainOrder)):
            saIdx = trainOrder[idx]
            subAgent = self.subAgents[saIdx]
            subAgent.inTraining = True
            subAgent.trainAgent = False
            numRuns = subAgent.decisionMaker.NumRuns()

            self.switchCounter[saIdx] = int(numRuns / self.numTrial2Switch)
            if self.switchCounter[saIdx] < minCounter:
                startAgent = idx
                minCounter = self.switchCounter[saIdx]

        self.currTrainSubAgentIdx = startAgent 

        self.SwitchTrainSubAgent()


    def SwitchTrainSubAgent(self, prevSaIdx = None):
        if prevSaIdx != None:
            prevSubAgent = self.subAgents[self.trainOrder[prevSaIdx]]
            prevSubAgent.trainAgent = False

            prevSubAgent.decisionMaker.decisionMaker.CopyTarget2DQN()

            self.currTrainSubAgentIdx = (self.currTrainSubAgentIdx + 1) % len(self.trainOrder)
         
        subAgentIdx = self.trainOrder[self.currTrainSubAgentIdx]
        currSubAgent = self.subAgents[subAgentIdx]
        numRuns = currSubAgent.decisionMaker.NumRuns()
        self.currCounterThreshold = int(numRuns / self.numTrial2Switch) * self.numTrial2Switch + self.numTrial2Switch
        currSubAgent.trainAgent = True

        self.decisionMaker.AddSwitch(subAgentIdx, self.switchCounter[subAgentIdx], SUBAGENTS_NAMES[self.trainOrder[self.currTrainSubAgentIdx]])
        self.switchCounter[subAgentIdx] += 1

        print("\n\nstart train sub agent:", SUBAGENTS_NAMES[self.trainOrder[self.currTrainSubAgentIdx]])
        print("counter threshold = ", self.currCounterThreshold)
        
    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1

        for key, sa in self.subAgents.items():
            if sa.FindActingHeirarchi() >= 0:
                return key
        
        return -1

    def FirstStep(self, obs):
        super(BaseMngr, self).FirstStep()

        # state       
        self.current_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')

        self.subAgentsActions = {}
        
        for sa in ALL_SUB_AGENTS:
            self.subAgentsActions[sa] = None
            self.subAgents[sa].FirstStep(obs) 

    def EndRun(self, reward, score, stepNum):
        if self.trainAgent or self.trainSubAgentsSimultaneously:
            self.decisionMaker.end_run(reward, score, stepNum)            

        for subAgent in self.subAgents.values():
            subAgent.EndRun(reward, score, stepNum)
        

        if self.trainSubAgentsSimultaneously:
            currSubAgent = self.subAgents[self.trainOrder[self.currTrainSubAgentIdx]]
            numRuns = currSubAgent.decisionMaker.NumRuns()

            if numRuns >= self.currCounterThreshold:
                self.SwitchTrainSubAgent(self.currTrainSubAgentIdx)


    def MonitorObservation(self, obs):
        for subAgent in self.subAgents.values():
            subAgent.MonitorObservation(obs)

    def Action2SC2Action(self, obs, a, moveNum):
        return self.subAgents[a].Action2SC2Action(obs, self.subAgentsActions[a], moveNum)
    
    def IsDoNothingAction(self, a):
        return a == ACTION_DO_NOTHING or self.subAgents[a].IsDoNothingAction(self.subAgentsActions[a])

    def CreateState(self, obs):
        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].CreateState(obs) 

        # retrieve state information from sub agents
        for key, value in BASE_STATE.STATE_IDX_MNGR_2_BUILDER_TRANSITIONS.items():
            self.current_state[key] = self.subAgents[SUB_AGENT_BUILDER].GetStateVal(value)

        for key, value in BASE_STATE.STATE_IDX_MNGR_2_TRAINER_TRANSITIONS.items():
            self.current_state[key] = self.subAgents[SUB_AGENT_TRAINER].GetStateVal(value)


        self.current_state[BASE_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[BASE_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]
        self.current_state[BASE_STATE.TIME_LINE] = self.sharedData.numStep

        self.ScaleState()

        self.sharedData.currBaseState = self.current_scaled_state.copy()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[BASE_STATE.MINERALS_IDX] = min(BASE_STATE.MINERALS_MAX, self.current_scaled_state[BASE_STATE.MINERALS_IDX])
        self.current_scaled_state[BASE_STATE.GAS_IDX] = min(BASE_STATE.GAS_MAX, self.current_scaled_state[BASE_STATE.GAS_IDX])

        self.current_scaled_state[BASE_STATE.SUPPLY_LEFT] = min(BASE_STATE.SUPPLY_LEFT_MAX, self.current_scaled_state[BASE_STATE.SUPPLY_LEFT])

    def Learn(self, reward, terminal):            
        if self.trainAgent:
            if self.isActionCommitted:
                self.decisionMaker.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                # if terminal reward entire state if action is not chosen for current step
                for a in range(NUM_ACTIONS):
                    self.decisionMaker.learn(self.previous_scaled_state, a, reward, self.terminalState, terminal)
                    self.decisionMaker.learn(self.current_scaled_state, a, reward, self.terminalState, terminal)

        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].Learn(reward, terminal) 

        self.previous_scaled_state[:] = self.current_scaled_state[:]
        self.isActionCommitted = False

    def SubAgentActionChosen(self, action):
        self.isActionCommitted = True
        self.lastActionCommitted = action

        if action in ALL_SUB_AGENTS:
            self.subAgents[action].SubAgentActionChosen(self.subAgentsActions[action])

    def ChooseAction(self):
        for sa in ALL_SUB_AGENTS:
            self.subAgentsActions[sa] = self.subAgents[sa].ChooseAction() 

        if self.playAgent:
            if self.illigalmoveSolveInModel:
                validActions = self.ValidActions()
                if self.trainAgent:
                    targetValues = False
                    exploreProb = self.decisionMaker.ExploreProb()              
                else:
                    targetValues = True
                    exploreProb = 1 #self.decisionMaker.TargetExploreProb()   

                if np.random.uniform() > exploreProb:
                    valVec = self.decisionMaker.ActionValuesVec(self.current_state, targetValues)   
                    random.shuffle(validActions)
                    validVal = valVec[validActions]
                    action = validActions[validVal.argmax()]
                else:
                    action = np.random.choice(validActions) 
            else:
                action = self.decisionMaker.choose_action(self.current_state)
        else:
            action = self.subAgentPlay


        self.current_action = action
        return action

    def ValidActions(self):
        valid = [ACTION_DO_NOTHING]
        if self.current_scaled_state[BASE_STATE.MINERALS_IDX] >= self.minPriceMinerals:
            valid.append(ACTION_BUILD_BASE)
        
            if self.ArmyBuildingExist():
                valid.append(ACTION_TRAIN_ARMY)
        
        return valid
    
    def ArmyBuildingExist(self):
        return (self.current_scaled_state[BASE_STATE.TRAIN_BUILDING_RELATED_IDX] > 0).any()

    def Action2Str(self, a):
        if a == ACTION_DO_NOTHING:
            return ACTION2STR[a]
        else:
            return ACTION2STR[a] + "-->" + self.subAgents[a].Action2Str(self.subAgentsActions[a])

    def PrintState(self):
        for i in range(BASE_STATE.SIZE):
            print(BASE_STATE.IDX2STR[i], self.current_scaled_state[i], end = ', ')
        print(']')

if __name__ == "__main__":
    if "results" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES)
    
    if "resultsSwitchingTrain" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, subAgentsGroups = list(SUBAGENTS_NAMES.values()))