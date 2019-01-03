import random
import math
import os.path
import datetime
import time
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


from pysc2.lib import actions
from pysc2.lib.units import Terran

from utils import BaseAgent

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

from algo_decisionMaker import CreateDecisionMaker

from utils_results import PlotResults

# sub agents
from agent_do_nothing import DoNothingSubAgent
from agent_base_mngr import BaseMngr
from agent_scout import ScoutAgent
from agent_attack import AttackAgent

# shared data
from agent_base_mngr import SharedDataBase
from agent_attack import SharedDataAttack
from agent_scout import SharedDataScout
from agent_do_nothing import SharedDataDoNothing

from agent_base_mngr import BASE_STATE
from agent_build_base import Building

from utils import SwapPnt
from utils import FindMiddle
from utils import Scale2MiniMap
from utils import GetScreenCorners

AGENT_NAME = "super_agent"

GRID_SIZE = 2

NUM_TRIALS_2_LEARN = 20

STEP_DURATION = 0


SUB_AGENT_ID_DONOTHING = 0
SUB_AGENT_ID_BASE = 1
SUB_AGENT_ID_ATTACK = 2
SUB_AGENT_ID_SCOUT = 3
NUM_SUB_AGENTS = 4

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_ID_DONOTHING] = "DoNothingSubAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_BASE] = "BaseMngr"
SUBAGENTS_NAMES[SUB_AGENT_ID_SCOUT] = "ScoutAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_ATTACK] = "AttackAgent"


# state details
class SUPER_STATE:
    BASE_DEVELOP_SCORE = 0
    ARMY_POWER = 1
    BASE_ACTION_ADVANTAGE = 2

    # power and fog mat
    SELF_POWER_START = 3
    SELF_POWER_END = SELF_POWER_START + GRID_SIZE * GRID_SIZE
    FOG_MAT_START = SELF_POWER_END
    FOG_MAT_END = FOG_MAT_START + GRID_SIZE * GRID_SIZE
    FOG_COUNTER_MAT_START = FOG_MAT_END
    FOG_COUNTER_MAT_END = FOG_COUNTER_MAT_START + GRID_SIZE * GRID_SIZE

    ENEMY_ARMY_MAT_START = FOG_COUNTER_MAT_END
    ENEMY_ARMY_MAT_END = ENEMY_ARMY_MAT_START + GRID_SIZE * GRID_SIZE

    TIME_LINE_IDX = ENEMY_ARMY_MAT_END

    SIZE = TIME_LINE_IDX + 1

    MAX_SCOUT_VAL = 10
    VAL_IS_SCOUTED = 8


class SUPER_ACTIONS:
    
    ACTION_DO_NOTHING = 0
    ACTION_DEVELOP_BASE = 1
    ACTION_SCOUT_START = 2
    ACTION_SCOUT_END = ACTION_SCOUT_START + GRID_SIZE * GRID_SIZE
    ACTION_ATTACK_START = ACTION_SCOUT_END
    ACTION_ATTACK_END = ACTION_ATTACK_START + GRID_SIZE * GRID_SIZE
    
    SIZE = ACTION_ATTACK_END

    ACTIONS2SUB_AGENTSID = {}
    ACTIONS2SUB_AGENTSID[ACTION_DO_NOTHING] = SUB_AGENT_ID_DONOTHING
    ACTIONS2SUB_AGENTSID[ACTION_DEVELOP_BASE] = SUB_AGENT_ID_BASE
    
    for a in range(ACTION_SCOUT_START, ACTION_SCOUT_END):
        ACTIONS2SUB_AGENTSID[a] = SUB_AGENT_ID_SCOUT

    for a in range(ACTION_ATTACK_START, ACTION_ATTACK_END):
        ACTIONS2SUB_AGENTSID[a] = SUB_AGENT_ID_ATTACK

    ACTION2STR = {}
    ACTION2STR[ACTION_DO_NOTHING] = "DoNothing"
    ACTION2STR[ACTION_DEVELOP_BASE] = "Develop_Base"
    for a in range(ACTION_SCOUT_START, ACTION_SCOUT_END):
        ACTION2STR[a] = "Scout_" + str(a - ACTION_SCOUT_START)

    for a in range(ACTION_ATTACK_START, ACTION_ATTACK_END):
        ACTION2STR[a] = "Attack_" + str(a - ACTION_ATTACK_START)


UNIT_VALUE_TABLE_NAME = 'unit_value_table.gz'

class SharedDataSuper(SharedDataBase, SharedDataAttack, SharedDataScout, SharedDataDoNothing):
    def __init__(self):
        super(SharedDataSuper, self).__init__()
        self.selfArmyMat = np.zeros((GRID_SIZE,GRID_SIZE), int)
        self.superGridSize = GRID_SIZE
        self.numStep = 0
        self.numAgentStep = 0

REWARD_MAX_SUPPLY = 0
BASE_MNGR_MAX_NAIVE_REWARD = 3.5
NORMALIZATION_TRAIN_LOCAL_REWARD = 2 * BASE_MNGR_MAX_NAIVE_REWARD

BATTLE_TYPES = set(["battle_mngr", "attack_army", "attack_base"])

class SuperAgent(BaseAgent):
    def __init__(self, configDict, useMapRewards=True, decisionMaker=None, isMultiThreaded=False, playList=[], trainList=[], testList=[], dmCopy=None):
        super(SuperAgent, self).__init__(SUPER_STATE.SIZE)

        self.sharedData = SharedDataSuper()

        self.trainAgent = AGENT_NAME in trainList
        self.testAgent = AGENT_NAME in testList
        self.playAgent = (AGENT_NAME in playList) | ("inherit" in playList)
        if self.playAgent:
            saPlayList = ["inherit"]
        else:
            saPlayList = playList
            if len(BATTLE_TYPES.intersection(playList)) == 0:
                saPlayList.append("do_nothing")
            
        self.illigalmoveSolveInModel = True

        if decisionMaker != None:
            self.decisionMaker = decisionMaker
        else:
            self.decisionMaker, _ = CreateDecisionMaker(agentName=AGENT_NAME, configDict=configDict, isMultiThreaded=isMultiThreaded, dmCopy=dmCopy)

        self.history = self.decisionMaker.AddHistory()
        
        # create sub agents and get decision makers
        self.subAgents = {}

        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)
            self.subAgents[key] = saClass(sharedData=self.sharedData, configDict=configDict, decisionMaker=saDM, isMultiThreaded=isMultiThreaded, 
                                                playList=saPlayList, trainList=trainList, testList=testList, dmCopy=dmCopy)
            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())

        if not self.playAgent:
            self.subAgentPlay = self.FindActingHeirarchi()
            self.activeSubAgents = [self.subAgentPlay]
            if self.subAgentPlay == SUB_AGENT_ID_BASE:
                self.activeSubAgents.append(SUB_AGENT_ID_DONOTHING)
        else:
            self.activeSubAgents = list(range(NUM_SUB_AGENTS))

        self.unitPower = {}
        table = pd.read_pickle(UNIT_VALUE_TABLE_NAME, compression='gzip')
        valVecMarine = table.ix['marine', :]
        self.unitPower[Terran.Marine] = sum(valVecMarine) / len(valVecMarine)
        self.unitPower[Terran.Reaper] = sum(table.ix['reaper', :]) / len(valVecMarine)
        self.unitPower[Terran.Hellion] = sum(table.ix['hellion', :]) / len(valVecMarine)
        self.unitPower[Terran.SiegeTank] = sum(table.ix['siege tank', :]) / len(valVecMarine)

        self.useMapRewards = useMapRewards

        self.move_number = 0
        self.discountForLocalReward = 0.999

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
        
        for sa in self.subAgents.values():
            ret = sa.GetAgentByName(name)
            if ret != None:
                return ret
            
        return None
     
    def GetDecisionMaker(self):
        return self.decisionMaker

    def FindActingHeirarchi(self):
        if self.playAgent:
            return 1

        saPlay = []
        for key, sa in self.subAgents.items():
            if sa.FindActingHeirarchi() >= 0:
                saPlay.append(key)

        if len(saPlay) > 0:
            return max(saPlay)
        
        return -1

    def step(self, obs):
        super(SuperAgent, self).step(obs)

        self.unit_type = obs.observation['feature_screen'][SC2_Params.UNIT_TYPE]
        if obs.first():
            self.FirstStep(obs)
            return self.SendAction(self.go2BaseAction)
        elif self.sharedData.numStep == 1:
            # get default screen location
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            self.baseNorthWestScreenCorner = cameraCornerNorthWest
        elif obs.last():
            self.LastStep(obs)
            return SC2_Actions.DO_NOTHING_SC2_ACTION

        time.sleep(STEP_DURATION)

        self.MonitorObservation(obs)
        if self.move_number == 0:
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            
            if cameraCornerNorthWest != self.baseNorthWestScreenCorner:
                return self.SendAction(self.go2BaseAction)

            self.sharedData.numAgentStep += 1
            self.CreateState(obs)
            self.Learn()
            self.ChooseAction()

            # self.PrintState()

        
        sc2Action = self.ActAction(obs)
        return self.SendAction(sc2Action)

    def FirstStep(self, obs):
        super(SuperAgent, self).FirstStep(obs)

        self.move_number = 0
        self.sharedData.numStep = 0 
        self.sharedData.numAgentStep = 0

        self.sharedData.__init__()
        
        cc_y, cc_x = (self.unit_type == Terran.CommandCenter).nonzero()
        if len(cc_y) > 0:
            middleCC = FindMiddle(cc_y, cc_x)
            cameraCornerNorthWest , cameraCornerSouthEast = GetScreenCorners(obs)
            miniMapLoc = Scale2MiniMap(middleCC, cameraCornerNorthWest, cameraCornerSouthEast)
            self.sharedData.commandCenterLoc = [miniMapLoc]
            self.sharedData.buildingCompleted[Terran.CommandCenter].append(Building(middleCC))

        self.sharedData.unitTrainValue = self.unitPower

        # actions:
        self.current_action = None

        # states:
        self.current_state = np.zeros(SUPER_STATE.SIZE, float)
        self.previous_scaled_state = np.zeros(SUPER_STATE.SIZE, float)
        self.current_scaled_state = np.zeros(SUPER_STATE.SIZE, float)

        self.accumulatedTrainReward = 0.0

        self.subAgentsActions = {}
        for sa in range(NUM_SUB_AGENTS):
            self.subAgentsActions[sa] = None
            self.subAgents[sa].FirstStep(obs)
        
        if len(self.sharedData.commandCenterLoc) >= 1:
            self.go2BaseAction = actions.FunctionCall(SC2_Actions.MOVE_CAMERA, [SwapPnt(self.sharedData.commandCenterLoc[0])])
        else:
            self.go2BaseAction = SC2_Actions.DO_NOTHING_SC2_ACTION

        self.maxSupply = False  

    def LastStep(self, obs):
        if self.useMapRewards:
            reward = obs.reward
        else:
            reward = 0

        sumReward = reward
        if SUB_AGENT_ID_BASE in self.activeSubAgents and not self.trainAgent:
            sumReward += self.accumulatedTrainReward
        
        score = obs.observation["score_cumulative"][0]

        self.CreateState(obs)
        
        self.AddTerminalReward(sumReward)

        self.Learn(sumReward, True)  
        self.EndRun(sumReward, score, self.sharedData.numAgentStep)

    def SendAction(self, sc2Action):
        self.sharedData.numStep += 1
        return sc2Action

    def EndRun(self, reward, score, numStep):
        if self.trainAgent or self.testAgent:
            self.decisionMaker.end_run(reward, score, numStep)
        
        for sa in self.activeSubAgents:
            self.subAgents[sa].EndRun(reward, score, numStep) 

    def ChooseAction(self):
        for sa in self.activeSubAgents:
                self.subAgentsActions[sa] = self.subAgents[sa].ChooseAction()

        if self.playAgent:
            validActions = self.ValidActions(self.current_scaled_state) if self.illigalmoveSolveInModel else list(range(SUPER_ACTIONS.SIZE))
            targetValues = False if self.trainAgent else True
            action, _ = self.decisionMaker.choose_action(self.current_scaled_state, validActions, targetValues)

        else:
            if self.subAgentPlay == SUB_AGENT_ID_ATTACK:
                action = SUPER_ACTIONS.ACTION_ATTACK_START
            else:
                action = self.subAgentPlay

        self.current_action = action
        return action
    
    def ValidActions(self, state):
        valid = [SUPER_ACTIONS.ACTION_DO_NOTHING, SUPER_ACTIONS.ACTION_DEVELOP_BASE]

        if state[SUPER_STATE.ARMY_POWER] > 0:
            for i in range(GRID_SIZE * GRID_SIZE):
                if state[SUPER_STATE.FOG_MAT_START + i] < SUPER_STATE.VAL_IS_SCOUTED:
                    valid.append(SUPER_ACTIONS.ACTION_SCOUT_START + i)
                
                valid.append(SUPER_ACTIONS.ACTION_ATTACK_START + i)

        return valid

    def Action2Str(self, realAct, onlyAgent=False):
        subAgent, subAgentAction = self.AdjustAction2SubAgents()
        if realAct:
            subAgent, subAgentAction = self.GetAction2Act(subAgent, subAgentAction)

        if onlyAgent:
            return SUBAGENTS_NAMES[subAgent]    
        else:
            return SUBAGENTS_NAMES[subAgent] + "-->" + self.subAgents[subAgent].Action2Str(subAgentAction)

    def ActAction(self, obs): 
        # get subagent and action
        subAgent, subAgentAction = self.AdjustAction2SubAgents()
        # mark to sub agent that his action was chosen
        self.subAgents[subAgent].SubAgentActionChosen(subAgentAction)
        # find real action to act
        subAgent, subAgentAction = self.GetAction2Act(subAgent, subAgentAction)
        # preform action
        sc2Action, terminal = self.subAgents[subAgent].Action2SC2Action(obs, subAgentAction, self.move_number)

        if terminal:
            self.move_number = 0
        else:
            self.move_number += 1

        return sc2Action


    def MonitorObservation(self, obs):
        for sa in self.activeSubAgents:
            self.subAgents[sa].MonitorObservation(obs)

        self.maxSupply = (obs.observation['player'][SC2_Params.SUPPLY_USED] == obs.observation['player'][SC2_Params.SUPPLY_CAP])
                     
    def CreateState(self, obs):

        for subAgent in self.subAgents.values():
            subAgent.CreateState(obs)

        if self.playAgent:
            baseMngrAgent = self.subAgents[SUB_AGENT_ID_BASE]
            self.current_state[SUPER_STATE.BASE_DEVELOP_SCORE] = baseMngrAgent.GetBaseDevelopScore()
            self.current_state[SUPER_STATE.ARMY_POWER] = baseMngrAgent.GetArmyPower()
            self.current_state[SUPER_STATE.BASE_ACTION_ADVANTAGE] = baseMngrAgent.GetMngrActionAdvantage()
            
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    idx = x + y * GRID_SIZE
                    self.current_state[idx + SUPER_STATE.SELF_POWER_START] = self.sharedData.selfArmyMat[y, x]
                    self.current_state[idx + SUPER_STATE.ENEMY_ARMY_MAT_START] = self.sharedData.enemyMatObservation[y, x]
                    self.current_state[idx + SUPER_STATE.FOG_MAT_START] = int( self.sharedData.fogRatioMat[y, x] * SUPER_STATE.MAX_SCOUT_VAL)
                    self.current_state[idx + SUPER_STATE.FOG_COUNTER_MAT_START] = self.sharedData.fogCounterMat[y, x]

            self.current_state[SUPER_STATE.TIME_LINE_IDX] = self.sharedData.numStep    
        
            self.ScaleCurrState()

    def AdjustAction2SubAgents(self):
        subAgentIdx = SUPER_ACTIONS.ACTIONS2SUB_AGENTSID[self.current_action]
    
        if subAgentIdx == SUB_AGENT_ID_SCOUT:
            self.subAgentsActions[subAgentIdx] = self.current_action - SUPER_ACTIONS.ACTION_SCOUT_START
        elif subAgentIdx == SUB_AGENT_ID_ATTACK:            
            self.subAgentsActions[subAgentIdx] = self.current_action - SUPER_ACTIONS.ACTION_ATTACK_START

        return subAgentIdx, self.subAgentsActions[subAgentIdx]

    def GetAction2Act(self, subAgentIdx, subAgentAction):  
        if SUB_AGENT_ID_DONOTHING in self.activeSubAgents and self.subAgents[subAgentIdx].IsDoNothingAction(subAgentAction):
            subAgentIdx = SUPER_ACTIONS.ACTION_DO_NOTHING
            subAgentAction = self.subAgentsActions[subAgentIdx]

        return subAgentIdx, subAgentAction

    def Learn(self, reward = 0, terminal = False):
        
        if SUB_AGENT_ID_BASE in self.activeSubAgents and not self.trainAgent:
            localReward = self.sharedData.prevTrainActionReward / NORMALIZATION_TRAIN_LOCAL_REWARD
            maxSupplyReward = REWARD_MAX_SUPPLY if self.maxSupply else 0.0 
            reward += localReward + maxSupplyReward

            self.accumulatedTrainReward += reward * pow(self.discountForLocalReward, self.sharedData.numAgentStep)
            self.sharedData.prevTrainActionReward = 0
         
        for sa in self.activeSubAgents:
            self.subAgents[sa].Learn(reward, terminal)
            
        if self.history != None and self.trainAgent and self.current_action != None:
            if terminal:
                reward = self.NormalizeReward(reward)
            self.history.learn(self.previous_scaled_state, self.current_action, reward, self.current_scaled_state, terminal)

        self.previous_scaled_state[:] = self.current_scaled_state[:]

     
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

    def PrintState(self):
        print ("base score =" , self.current_scaled_state[SUPER_STATE.BASE_DEVELOP_SCORE], 
        "base action advantage =", self.current_scaled_state[SUPER_STATE.BASE_ACTION_ADVANTAGE], 
        "army power =", self.current_scaled_state[SUPER_STATE.ARMY_POWER])

        # print("barracks: min =",  self.current_state[STATE.BASE.QUEUE_BARRACKS], end = '')
        # barList = self.sharedData.buildingCompleted[Terran.Barracks] 
        # idx = 0
        # numInQ = 0

        # for b in barList:
        #     idx += 1
        #     print("\nbarracks", idx, end = ": ")
        #     for u in b.qForProduction:
        #         numInQ += 1
        #         print(TerranUnit.ARMY_SPEC[u.unit].name, u.step, end = ', ')

        # idx = 0
        # for b in self.sharedData.buildingCompleted[Terran.BarracksReactor]:
        #     idx += 1
        #     print("\nreactor", idx, end = ": ")
        #     for u in b.qForProduction:
        #         numInQ += 1
        #         print(TerranUnit.ARMY_SPEC[u.unit].name, u.step, end = ', ')


        # print("\nfactory: fq:", self.current_state[STATE.BASE.QUEUE_FACTORY], "tlq:", self.current_state[STATE.BASE.QUEUE_TECHLAB], end = '')
        # faList = self.sharedData.buildingCompleted[Terran.Factory] 
        # idx = 0
        # for f in faList:
        #     idx += 1
        #     print("\nfactory:", idx, end = ": ")
        #     for u in f.qForProduction:
        #         numInQ += 1
        #         print(TerranUnit.ARMY_SPEC[u.unit].name, u.step, end = ', ')

        # idx = 0
        # for f in self.sharedData.buildingCompleted[Terran.FactoryTechLab]:
        #     idx += 1
        #     print("\ntechlab", idx, end = ": ")
        #     for u in f.qForProduction:
        #         numInQ += 1
        #         print(TerranUnit.ARMY_SPEC[u.unit].name, u.step, end = ', ')

        # print("\n\n")

        # self.subAgents[SUB_AGENT_ID_BASE].PrintState()

        # print("self mat     fog mat     enemy mat")
        # for y in range(STATE.GRID_SIZE):
        #     for x in range(STATE.GRID_SIZE):
        #         idx = x + y * STATE.GRID_SIZE
        #         print(self.current_scaled_state[STATE.SELF_POWER_START + idx], end = ' ')

        #     print(end = "   |   ")
        #     for x in range(STATE.GRID_SIZE):
        #         idx = x + y * STATE.GRID_SIZE
        #         print(self.current_scaled_state[STATE.FOG_MAT_START + idx], end = ',')
        #         print(self.current_scaled_state[STATE.FOG_COUNTER_MAT_START + idx], end = ' ')

        #     print(end = "   |   ")
        #     for x in range(STATE.GRID_SIZE):
        #         idx = x + y * STATE.GRID_SIZE
        #         print(self.current_scaled_state[STATE.ENEMY_ARMY_MAT_START + idx], end = ' ')

        #     print("||")
  

if __name__ == "__main__":
    from absl import app
    from absl import flags
    flags.DEFINE_string("directoryPrefix", "", "directory names to take results")
    flags.DEFINE_string("directoryNames", "", "directory names to take results")
    flags.DEFINE_string("grouping", "100", "grouping size of results.")
    flags.DEFINE_string("max2Plot", "none", "grouping size of results.")
    flags.FLAGS(sys.argv)

    directoryNames = (flags.FLAGS.directoryNames).split(",")
    for d in range(len(directoryNames)):
        directoryNames[d] = flags.FLAGS.directoryPrefix + directoryNames[d]
    
    grouping = int(flags.FLAGS.grouping)
    if flags.FLAGS.max2Plot == "none":
        max2Plot = None
    else:
        max2Plot = int(flags.FLAGS.max2Plot)

    if "results" in sys.argv:
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot)
    elif "multipleResults" in sys.argv:
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot, multipleDm=True)
