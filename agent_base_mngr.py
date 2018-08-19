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
from utils_decisionMaker import BaseDecisionMaker

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
from agent_resource_gather import SharedDataGather

from utils import GetScreenCorners
from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap

AGENT_DIR = "BaseMngr/"
AGENT_NAME = "base_mngr"


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

class SharedDataBase(SharedDataBuild, SharedDataTrain, SharedDataGather):
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
    IN_PROGRESS_RECTORS_IDX = 13
    IN_PROGRESS_TECHLAB_IDX = 14    

    ARMY_POWER = 15
    QUEUE_BARRACKS = 16
    QUEUE_FACTORY = 17
    QUEUE_TECHLAB = 18

    SIZE = 19

    BUILDING_RELATED_IDX = [COMMAND_CENTER_IDX] + list(range(SUPPLY_DEPOT_IDX, ARMY_POWER))
    TRAIN_BUILDING_RELATED_IDX = [BARRACKS_IDX, FACTORY_IDX]

    BUILDING_2_STATE_TRANSITION = {}
    BUILDING_2_STATE_TRANSITION[Terran.CommandCenter] = [COMMAND_CENTER_IDX, -1]
    BUILDING_2_STATE_TRANSITION[Terran.SupplyDepot] = [SUPPLY_DEPOT_IDX, IN_PROGRESS_SUPPLY_DEPOT_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Refinery] = [REFINERY_IDX, IN_PROGRESS_REFINERY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Barracks] = [BARRACKS_IDX, IN_PROGRESS_BARRACKS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Factory] = [FACTORY_IDX, IN_PROGRESS_FACTORY_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.Reactor] = [REACTORS_IDX, IN_PROGRESS_RECTORS_IDX]
    BUILDING_2_STATE_TRANSITION[Terran.TechLab] = [TECHLAB_IDX, IN_PROGRESS_TECHLAB_IDX]

    BUILDING_2_STATE_QUEUE_TRANSITION = {}

    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Barracks] = QUEUE_BARRACKS
    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.Factory] = QUEUE_FACTORY
    BUILDING_2_STATE_QUEUE_TRANSITION[Terran.TechLab] = QUEUE_TECHLAB


    IDX2STR = ["CC", "MIN", "GAS", "SD", "REF", "BA", "FA", "REA", "TECH", "SD_B", "REF_B", "BA_B", "FA_B", "REA_B", "TECH_B", "POWER", "BA_Q", "FA_Q", "TE_Q"]

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



DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

class NaiveDecisionMakerBaseMngr(BaseDecisionMaker):
    def __init__(self, resultFName = None, directory = None, numTrials2Learn = 20):
        super(NaiveDecisionMakerBaseMngr, self).__init__()
        self.resultFName = resultFName
        self.trialNum = 0
        self.numTrials2Learn = numTrials2Learn

        if resultFName != None:
            self.lock = Lock()
            if directory != None:
                fullDirectoryName = "./" + directory +"/"
            else:
                fullDirectoryName = "./"

            if "new" in sys.argv:
                loadFiles = False
            else:
                loadFiles = True

            self.resultFile = ResultFile(fullDirectoryName + resultFName, numTrials2Learn, loadFiles)

    def end_run(self, r, score, steps):
        saveFile = False
        self.trialNum += 1
        
        if self.resultFName != None:
            self.lock.acquire()
            if self.trialNum % self.numTrials2Learn == 0:
                saveFile = True

            self.resultFile.end_run(r, score, steps, saveFile)
            self.lock.release()
       
        return saveFile

    def choose_action(self, state):
        action = 0

        numSDAll = state[BASE_STATE.SUPPLY_DEPOT_IDX] + state[BASE_STATE.IN_PROGRESS_SUPPLY_DEPOT_IDX]
        numRefAll = state[BASE_STATE.REFINERY_IDX] + state[BASE_STATE.IN_PROGRESS_REFINERY_IDX]
        numBaAll = state[BASE_STATE.BARRACKS_IDX] + state[BASE_STATE.IN_PROGRESS_BARRACKS_IDX]
        numFaAll = state[BASE_STATE.FACTORY_IDX] + state[BASE_STATE.IN_PROGRESS_FACTORY_IDX]
        numReactorsAll = state[BASE_STATE.REACTORS_IDX] + state[BASE_STATE.IN_PROGRESS_RECTORS_IDX]
        numTechAll = state[BASE_STATE.TECHLAB_IDX] + state[BASE_STATE.IN_PROGRESS_TECHLAB_IDX]
        numBarracksQ = state[BASE_STATE.QUEUE_BARRACKS]
        
        power = state[BASE_STATE.ARMY_POWER]

        if numSDAll < 3 or numRefAll < 2 or numBaAll < 2 or numReactorsAll < 2:
            action = ACTION_BUILD_BASE
        elif numBarracksQ < 6 and power < 6:
            action = ACTION_TRAIN_ARMY
        elif numFaAll < 2: 
            action = ACTION_BUILD_BASE
        elif power < 10:
            action = ACTION_TRAIN_ARMY
        elif numTechAll < 2:
            action = ACTION_BUILD_BASE
        elif power < 30:
            action = ACTION_TRAIN_ARMY
        elif numSDAll < 6:
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

        self.sharedData = sharedData
        self.terminalState = np.zeros(BASE_STATE.SIZE, dtype=np.int, order='C')


        # model params 
        self.minPriceMinerals = 50

    def CreateDecisionMaker(self, dmTypes, isMultiThreaded):
        
        runType = RUN_TYPES[dmTypes[AGENT_NAME]]
        # create agent dir
        directory = dmTypes["directory"] + "/" + AGENT_DIR
        if not os.path.isdir("./" + directory):
            os.makedirs("./" + directory)

        if dmTypes[AGENT_NAME] == "naive":
            decisionMaker = NaiveDecisionMakerBaseMngr(resultFName=runType[RESULTS], directory=directory + runType[DIRECTORY])
        else:        
            decisionMaker = LearnWithReplayMngr(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME],  
                                                resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory + runType[DIRECTORY], isMultiThreaded=isMultiThreaded)

        return decisionMaker

    def GetDecisionMaker(self):
        return self.decisionMaker

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
        if self.trainAgent:
            self.decisionMaker.end_run(reward, score, stepNum)
        
        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].EndRun(reward, score, stepNum)

    def Action2SC2Action(self, obs, a, moveNum):
        return self.subAgents[a].Action2SC2Action(obs, self.subAgentsActions[a], moveNum)
    
    def IsDoNothingAction(self, a):
        return a == ACTION_DO_NOTHING or self.subAgents[a].IsDoNothingAction(self.subAgentsActions[a])
        
    def CreateState(self, obs):

        for sa in ALL_SUB_AGENTS:
            self.subAgents[sa].CreateState(obs) 

        for key, value in BASE_STATE.BUILDING_2_STATE_TRANSITION.items():
            self.current_state[value[0]] = self.sharedData.buildingCount[key]
            if value[1] >= 0:
                self.current_state[value[1]] = len(self.sharedData.buildCommands[key])

        self.current_state[BASE_STATE.MINERALS_IDX] = obs.observation['player'][SC2_Params.MINERALS]
        self.current_state[BASE_STATE.GAS_IDX] = obs.observation['player'][SC2_Params.VESPENE]

        for key, value in BASE_STATE.BUILDING_2_STATE_QUEUE_TRANSITION.items():
            self.current_state[value] = len(self.sharedData.trainingQueue[key])
        
        power = 0
        for unit, num in self.sharedData.armySize.items():
            power += num * self.sharedData.unitTrainValue[unit]

        self.current_state[BASE_STATE.ARMY_POWER] = power

        self.ScaleState()

        self.sharedData.currBaseState = self.current_scaled_state.copy()

    def ScaleState(self):
        self.current_scaled_state[:] = self.current_state[:]
        
        self.current_scaled_state[BASE_STATE.MINERALS_IDX] = int(self.current_scaled_state[BASE_STATE.MINERALS_IDX] / BASE_STATE.MINERALS_BUCKETING) * BASE_STATE.MINERALS_BUCKETING
        self.current_scaled_state[BASE_STATE.MINERALS_IDX] = min(BASE_STATE.MINERALS_MAX, self.current_scaled_state[BASE_STATE.MINERALS_IDX])
        self.current_scaled_state[BASE_STATE.GAS_IDX] = int(self.current_scaled_state[BASE_STATE.GAS_IDX] / BASE_STATE.GAS_BUCKETING) * BASE_STATE.GAS_BUCKETING
        self.current_scaled_state[BASE_STATE.GAS_IDX] = min(BASE_STATE.GAS_MAX, self.current_scaled_state[BASE_STATE.GAS_IDX])

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

    def NewBuildings(self):
        return (self.previous_scaled_state[BASE_STATE.BUILDING_RELATED_IDX] != self.current_scaled_state[BASE_STATE.BUILDING_RELATED_IDX]).any()
    
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