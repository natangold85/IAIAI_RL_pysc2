import random
import math
import os.path
import sys
import logging
import traceback
import datetime
from multiprocessing import Lock

import threading

import numpy as np
import pandas as pd
import time

from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from utils_decisionMaker import BaseDecisionMaker
from utils_decisionMaker import DecisionMakerMngr
from utils_decisionMaker import UserPlay
from utils_decisionMaker import BaseNaiveDecisionMaker

from utils_results import PlotResults
from utils_results import ResultFile


from utils_qtable import QTableParamsExplorationDecay
from utils_dqn import DQN_PARAMS
from utils_dqn import DQN_PARAMS_WITH_DEFAULT_DM
from utils_a3c import A3C_PARAMS

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
TRAIN_ALL = "base_mngr_all"

ACTION_DO_NOTHING = 0
ACTION_BUILD_BASE = 1
ACTION_TRAIN_ARMY = 2
NUM_ACTIONS = 3

SUB_AGENT_BUILDER = 1
SUB_AGENT_TRAINER = 2
ID_SELF_AGENT = 3
ALL_SUB_AGENTS = [SUB_AGENT_BUILDER, SUB_AGENT_TRAINER]

ACTION2STR = ["DoNothing", "BuildBase", "TrainArmy"]

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_BUILDER] = "BuildBaseSubAgent"
SUBAGENTS_NAMES[SUB_AGENT_TRAINER] = "TrainArmySubAgent"

# Model Params
NUM_TRIALS_2_LEARN = 20
NUM_TRIALS_4_CMP = 200

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

    SUPPLY_LEFT_MAX = 20

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
    TIME_LINE_IDX = 20
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
    IDX2STR[TIME_LINE_IDX] = "TimeLine"

# possible types of play

QTABLE = 'q'
DQN = 'dqn'
DQN2L = 'dqn_2l'
DQN2L_DFLT = 'dqn_2l_dflt'
A3C = "A3C"
USER_PLAY = 'play'
NAIVE = 'naive'

# data for run type
TYPE = "type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "hist"
RESULTS = "results"
ALL_RESULTS = "all_results"
PARAMS = 'params'
DIRECTORY = 'directory'

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
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(BASE_STATE.SIZE, NUM_ACTIONS, numTrials2CmpResults=NUM_TRIALS_4_CMP)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "baseMngr_dqn"
RUN_TYPES[DQN][DIRECTORY] = "baseMngr_dqn"
RUN_TYPES[DQN][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[DQN][RESULTS] = "baseMngr_result"

RUN_TYPES[DQN2L] = {}
RUN_TYPES[DQN2L][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN2L][PARAMS] = DQN_PARAMS(BASE_STATE.SIZE, NUM_ACTIONS, layersNum=2, numTrials2CmpResults=NUM_TRIALS_4_CMP, descendingExploration=False)
RUN_TYPES[DQN2L][DECISION_MAKER_NAME] = "baseMngr_dqn2l"
RUN_TYPES[DQN2L][DIRECTORY] = "baseMngr_dqn2l"
RUN_TYPES[DQN2L][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[DQN2L][RESULTS] = "baseMngr_result"
RUN_TYPES[DQN2L][ALL_RESULTS] = "baseMngr_results_all"

RUN_TYPES[DQN2L_DFLT] = {}
RUN_TYPES[DQN2L_DFLT][TYPE] = "DQN_WithTargetAndDefault"
RUN_TYPES[DQN2L_DFLT][PARAMS] = DQN_PARAMS_WITH_DEFAULT_DM(BASE_STATE.SIZE, NUM_ACTIONS, layersNum=2, numTrials2CmpResults=NUM_TRIALS_4_CMP, descendingExploration = False)
RUN_TYPES[DQN2L_DFLT][DECISION_MAKER_NAME] = "baseMngr_dqn2l_dflt"
RUN_TYPES[DQN2L_DFLT][DIRECTORY] = "baseMngr_dqn2l_dflt"
RUN_TYPES[DQN2L_DFLT][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[DQN2L_DFLT][RESULTS] = "baseMngr_result"
RUN_TYPES[DQN2L_DFLT][ALL_RESULTS] = "baseMngr_results_all"

RUN_TYPES[A3C] = {}
RUN_TYPES[A3C][TYPE] = "A3C"
RUN_TYPES[A3C][PARAMS] = A3C_PARAMS(BASE_STATE.SIZE, NUM_ACTIONS, numTrials2CmpResults=NUM_TRIALS_4_CMP)
RUN_TYPES[A3C][DECISION_MAKER_NAME] = "baseMngr_A3C"
RUN_TYPES[A3C][DIRECTORY] = "baseMngr_A3C"
RUN_TYPES[A3C][HISTORY] = "baseMngr_replayHistory"
RUN_TYPES[A3C][RESULTS] = "baseMngr_result"
RUN_TYPES[A3C][ALL_RESULTS] = "baseMngr_results_all"

RUN_TYPES[NAIVE] = {}
RUN_TYPES[NAIVE][DIRECTORY] = "baseMngr_naive"
RUN_TYPES[NAIVE][RESULTS] = "baseMngr_result"
RUN_TYPES[NAIVE][ALL_RESULTS] = "baseMngr_AllResults"

def CreateDecisionMakerBaseMngr(dmTypes, isMultiThreaded, numTrials2Learn=NUM_TRIALS_2_LEARN):
    if dmTypes[AGENT_NAME] == "none":
        return BaseDecisionMaker(AGENT_NAME), []

    runType = RUN_TYPES[dmTypes[AGENT_NAME]]
    directory = dmTypes["directory"] + "/" + AGENT_DIR + runType[DIRECTORY]

    if dmTypes[AGENT_NAME] == "naive":
        decisionMaker = NaiveDecisionMakerBaseMngr(resultFName=runType[RESULTS], directory=directory)
    else:        
        if runType[TYPE] == "DQN_WithTargetAndDefault":
            runType[PARAMS].defaultDecisionMaker = NaiveDecisionMakerBaseMngr()

        decisionMaker = DecisionMakerMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], agentName=AGENT_NAME,  
                            resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory, isMultiThreaded=isMultiThreaded,
                            numTrials2Learn=numTrials2Learn)

    return decisionMaker, runType


class NaiveDecisionMakerBaseMngr(BaseNaiveDecisionMaker):
    def __init__(self, resultFName = None, directory=None, numTrials2Save =NUM_TRIALS_2_LEARN):
        super(NaiveDecisionMakerBaseMngr, self).__init__(numTrials2Save=numTrials2Save, agentName=AGENT_NAME, resultFName=resultFName, directory=directory)

    def choose_action(self, state, validActions, targetValues=False):
        action = ACTION_DO_NOTHING
        if np.random.uniform() > 0.75:
            return action

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

        return action if action in validActions else ACTION_DO_NOTHING

    def ActionsValues(self, state, validActions, target = True):    
        vals = np.zeros(NUM_ACTIONS,dtype = float)
        vals[self.choose_action(state, validActions)] = 1.0

        return vals

    def end_run(self, r, score, steps):
        print(threading.current_thread().getName(), ":", AGENT_NAME,"->for trial#", self.trialNum, ": reward =", r, "score =", score, "steps =", steps)
        return super(NaiveDecisionMakerBaseMngr, self).end_run(r, score, steps)


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
        self.trainAll = TRAIN_ALL in trainList

        self.inTraining = self.trainAgent or self.trainAll

        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
            self.allResultFile = self.decisionMaker.GetResultFile()
        else:
            self.decisionMaker = self.CreateDecisionMaker(dmTypes, isMultiThreaded)

        decisionMakerType = self.decisionMaker.DecisionMakerType()
        self.initialDfltDecisionMaker = decisionMakerType.find("Default") > 0
        self.isA3CAlg = decisionMakerType == "A3C"

        self.history = self.decisionMaker.AddHistory()

        # create sub agents and get decision makers
        self.subAgents = {}

        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)     
            self.subAgents[key] = saClass(sharedData, dmTypes, saDM, isMultiThreaded, saPlayList, trainList)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())


        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()

        self.currCounterThreshold = -1

        if self.trainAll:
            trainOrder = [SUB_AGENT_BUILDER, SUB_AGENT_TRAINER, SUB_AGENT_BUILDER, SUB_AGENT_TRAINER, ID_SELF_AGENT]
            self.SetSubAgentTrainSwitch(dmTypes["numTrial2Train"], trainOrder)
            
        elif self.trainSubAgentsSimultaneously:
            self.SetSubAgentTrainSwitch(dmTypes["numTrial2Train"], [SUB_AGENT_BUILDER, SUB_AGENT_TRAINER])

        self.sharedData = sharedData
        self.terminalState = np.zeros(BASE_STATE.SIZE, dtype=np.int, order='C')

        self.subAgentsActions = {}

        # model params 
        self.minPriceMinerals = 50

        self.current_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(BASE_STATE.SIZE, dtype=np.int32, order='C')



    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        decisionMaker, runType = CreateDecisionMakerBaseMngr(dmTypes, isMultiThreaded)
        if dmTypes[AGENT_NAME] == "none":
            return decisionMaker

        directory = dmTypes["directory"] + "/" + AGENT_DIR + runType[DIRECTORY]
        
        fullDirectoryName = "./" + directory +"/"
        if ALL_RESULTS in runType:
            loadFiles = False if "new" in sys.argv else True
            self.allResultFile = ResultFile(fullDirectoryName + runType[ALL_RESULTS], NUM_TRIALS_2_LEARN, loadFiles, "All_" + AGENT_NAME)
        else:
            self.allResultFile = None
        
        decisionMaker.AddResultFile(self.allResultFile)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
        
        for sa in self.subAgents.values():
            ret = sa.GetAgentByName(name)
            if ret != None:
                return ret
            
        return None

    def SetSubAgentTrainSwitch(self, numTrial2Switch, trainOrder):
        self.numTrial2Switch = numTrial2Switch
        self.trainOrder = trainOrder
        self.switchCounter = {}
        
        startAgent = 0
        minCounter = 1000
        for idx in range(len(trainOrder)):
            saIdx = trainOrder[idx]
            if saIdx == ID_SELF_AGENT:
                agent = self
            else:
                agent = self.subAgents[saIdx]

            agent.inTraining = True
            agent.trainAgent = False
            numRuns = agent.decisionMaker.NumRuns()

            self.switchCounter[saIdx] = int(numRuns / self.numTrial2Switch)
            if self.switchCounter[saIdx] < minCounter:
                startAgent = idx
                minCounter = self.switchCounter[saIdx]

        if self.initialDfltDecisionMaker and not self.decisionMaker.DfltValueInitialized():
            self.currTrainAgentIdx = -1
        else:
            self.currTrainAgentIdx = startAgent 

        self.SwitchTrainSubAgent()


    def SwitchTrainSubAgent(self, prevAgentIdx = None):
        if prevAgentIdx != None:
            if prevAgentIdx == ID_SELF_AGENT:
                prevAgent = self
            else:
                prevAgent = self.subAgents[prevAgentIdx]
            
            prevAgent.trainAgent = False
            prevAgent.decisionMaker.TakeTargetDM(self.currCounterThreshold)

            self.currTrainAgentIdx = (self.currTrainAgentIdx + 1) % len(self.trainOrder)
        
        # if idx is negative than all dm will take target values
        if self.currTrainAgentIdx >= 0:
            agentIdx = self.trainOrder[self.currTrainAgentIdx]
            if agentIdx == ID_SELF_AGENT:
                currAgent = self
                name = AGENT_NAME
            else:
                currAgent = self.subAgents[agentIdx]  
                name = SUBAGENTS_NAMES[agentIdx]

            numRuns = currAgent.decisionMaker.NumRuns()
            currAgent.decisionMaker.ResetHistory()
            self.currCounterThreshold = int(numRuns / self.numTrial2Switch) * self.numTrial2Switch + self.numTrial2Switch
            currAgent.trainAgent = True

            self.decisionMaker.AddSwitch(agentIdx, self.switchCounter[agentIdx], name, self.allResultFile)
            self.switchCounter[agentIdx] += 1

            print("\n\nstart train sub agent:", name)
            print("counter threshold = ", self.currCounterThreshold)
        
        else:
            self.decisionMaker.AddSwitch(self.currTrainAgentIdx, 0, "DfltVals", self.allResultFile)
            self.currCounterThreshold = self.decisionMaker.decisionMaker.trialsOfDfltRun
            
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

        self.lastActionCommittedStep = 0
        self.lastActionCommittedState = None
        self.lastActionCommittedNextState = None

        for sa in ALL_SUB_AGENTS:
            self.subAgentsActions[sa] = None
            self.subAgents[sa].FirstStep(obs) 

        # switch among agents if necessary
        if self.trainSubAgentsSimultaneously or self.trainAll:
            if self.currTrainAgentIdx >= 0:
                agentIdx = self.trainOrder[self.currTrainAgentIdx]
                
                if agentIdx == ID_SELF_AGENT:
                    numRuns = self.decisionMaker.NumRuns()
                else:
                    currAgent = self.subAgents[agentIdx]
                    numRuns = currAgent.decisionMaker.NumRuns()

                if numRuns >= self.currCounterThreshold:
                    self.SwitchTrainSubAgent(agentIdx)
            else:

                if self.decisionMaker.NumDfltRuns() >= self.currCounterThreshold:
                    self.currTrainAgentIdx = 0
                    self.SwitchTrainSubAgent()

    def EndRun(self, reward, score, stepNum):
        saveTables = False
        if self.trainAgent:
            saved = self.decisionMaker.end_run(reward, score, stepNum)
            saveTables |= saved  

        for subAgent in self.subAgents.values():
            saved = subAgent.EndRun(reward, score, stepNum)
            saveTables |= saved  
        
        if self.trainSubAgentsSimultaneously or self.trainAll:
            if self.allResultFile != None:
                self.allResultFile.end_run(reward, score, stepNum, saveTables)
                # # for default run (all agents with target values)
            if self.currTrainAgentIdx < 0:
                self.decisionMaker.end_run(reward, score, stepNum)
                for subAgent in self.subAgents.values():
                    subAgent.decisionMaker.end_run(reward, score, stepNum)

        return saveTables


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
        self.current_state[BASE_STATE.TIME_LINE_IDX] = self.sharedData.numStep

        self.ScaleState()

        self.sharedData.currBaseState = self.current_scaled_state.copy()

        if self.isActionCommitted:
            self.lastActionCommittedStep = self.sharedData.numAgentStep
            self.lastActionCommittedState = self.previous_scaled_state
            self.lastActionCommittedNextState = self.current_scaled_state

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[BASE_STATE.MINERALS_IDX] = min(BASE_STATE.MINERALS_MAX, self.current_scaled_state[BASE_STATE.MINERALS_IDX])
        self.current_scaled_state[BASE_STATE.GAS_IDX] = min(BASE_STATE.GAS_MAX, self.current_scaled_state[BASE_STATE.GAS_IDX])

        self.current_scaled_state[BASE_STATE.SUPPLY_LEFT] = min(BASE_STATE.SUPPLY_LEFT_MAX, self.current_scaled_state[BASE_STATE.SUPPLY_LEFT])

    def Learn(self, reward, terminal): 
        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].Learn(reward, terminal) 
            
        if self.history != None and self.trainAgent:
            reward = reward if not terminal else self.NormalizeReward(reward)

            if self.isActionCommitted:
                self.history.learn(self.previous_scaled_state, self.lastActionCommitted, reward, self.current_scaled_state, terminal)
            elif terminal:
                if self.lastActionCommitted != None:
                    numSteps = self.lastActionCommittedStep - self.sharedData.numAgentStep
                    discountedReward = reward * pow(self.decisionMaker.DiscountFactor(), numSteps)
                    self.history.learn(self.lastActionCommittedState, self.lastActionCommitted, discountedReward, self.lastActionCommittedNextState, terminal) 

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
            else: 
                validActions = list(range(NUM_ACTIONS))
 
            targetValues = False if self.trainAgent else True
            action = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)
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

    def Action2Str(self, a, onlyAgent=False):
        if a == ACTION_DO_NOTHING or a not in self.subAgentsActions.keys() or onlyAgent:
            return ACTION2STR[a]
        else:
            return ACTION2STR[a] + "-->" + self.subAgents[a].Action2Str(self.subAgentsActions[a])

    def StateIdx2Str(self, idx):
        return BASE_STATE.IDX2STR[idx]

    def PrintState(self):
        for i in range(BASE_STATE.SIZE):
            print(BASE_STATE.IDX2STR[i], self.current_scaled_state[i], end = ', ')
        print(']')


if __name__ == "__main__":
    from absl import app
    from absl import flags
    flags.DEFINE_string("directoryNames", "", "directory names to take results")
    flags.DEFINE_string("grouping", "100", "grouping size of results.")
    flags.FLAGS(sys.argv)

    directoryNames = (flags.FLAGS.directoryNames).split(",")
    grouping = int(flags.FLAGS.grouping)

    if "results" in sys.argv:
        print(directoryNames)
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping)
    
    if "resultsSwitchingTrain" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping, 
                    subAgentsGroups=list(SUBAGENTS_NAMES.values()) + [AGENT_NAME, "DfltVals"], keyResults=ALL_RESULTS, additionPlots=list(SUBAGENTS_NAMES.values()) + [AGENT_NAME])