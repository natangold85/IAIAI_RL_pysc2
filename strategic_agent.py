import random
import math
import os.path
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

from utils import SwapPnt

from utils_tables import TableMngr
from utils_tables import QTableParamsWOChangeInExploration

from build_base import BuildBaseSubAgent
from train_army import TrainArmySubAgent
from attack import AttackSubAgent

Q_TABLE_STRATEGIC_FILE = 'strategic_q_table'
Q_TABLE_BUILDBASE_FILE = 'buildbase_strategic_q_table'
Q_TABLE_TRAINARMY_FILE = 'trainarmy_strategic_q_table'
Q_TABLE_ATTACK_FILE = 'attack_strategic_q_table'

ID_ACTION_DO_NOTHING = 0
ID_ACTION_BUILD_BASE = 1
ID_ACTION_TRAIN_ARMY = 2
ID_ACTION_ATTACK = 3

NUM_ACTIONS = 4

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

STEP_DURATION = 0

# state details
STATE_NON_VALID_NUM = -1

STATE_MINERALS_MAX = 500
STATE_GAS_MAX = 300
STATE_MINERALS_BUCKETING = 50
STATE_GAS_BUCKETING = 50

STATE_COMMAND_CENTER_IDX = 0
STATE_SUPPLY_DEPOT_IDX = 1
STATE_REFINERY_IDX = 2
STATE_BARRACKS_IDX = 3
STATE_FACTORY_IDX = 4
STATE_ARMY_IDX = 5
STATE_ENEMYPOWER_IDX = 6
STATE_SIZE = 7


REWARD_TRAIN_MARINE = 1
REWARD_TRAIN_REAPER = 2
REWARD_TRAIN_HELLION = 4
REWARD_TRAIN_SIEGE_TANK = 6
NORMALIZED_REWARD = 300

CREATE_DETAILED_STATE = False
DETAILED_STATE_SIZE = STATE_SIZE

class Strategic(base_agent.BaseAgent):
    def __init__(self):
        super(Strategic, self).__init__()
        
        # sub agents
        self.m_buildBaseSubAgent = BuildBaseSubAgent(Q_TABLE_BUILDBASE_FILE)
        self.m_trainArmySubAgent = TrainArmySubAgent(Q_TABLE_TRAINARMY_FILE)
        self.m_attackSubAgent = AttackSubAgent(Q_TABLE_ATTACK_FILE)


        self.tables = TableMngr(NUM_ACTIONS, STATE_SIZE, Q_TABLE_STRATEGIC_FILE)

        # states and action:
        self.current_action = None
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.current_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        # decision maker
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.current_state_for_decision_making = None

        # model params 
        self.move_number = 0
        self.step_num = 0
        self.IsMultiSelect = False
               
        # for developing:
        self.bool = False
        self.coord = [-1,-1]

    def step(self, obs):
        super(Strategic, self).step(obs)
        self.step_num += 1
        
        try:
            self.unit_type = obs.observation['screen'][SC2_Params.UNIT_TYPE]
            
            if obs.last():
                self.LastStep(obs)
                return DO_NOTHING_SC2_ACTION
            elif obs.first():
                self.FirstStep(obs)
            
            time.sleep(STEP_DURATION)
            self.m_buildBaseSubAgent.UpdateBuildingInProgress(obs)
            if self.IsMultiSelect:
                self.m_buildBaseSubAgent.UpdateBuildingCompletion(obs)
                self.IsMultiSelect = False

            if self.move_number == 0:
                self.CreateState(obs)
                self.Learn(obs)

                self.current_action = self.tables.choose_action(str(self.current_scaled_state))
                self.previous_state[:] = self.current_state[:]
                self.previous_scaled_state[:] = self.current_scaled_state[:]

            action2Return = DO_NOTHING_SC2_ACTION
            if self.current_action == ID_ACTION_DO_NOTHING:
                # print("do nothing action move num =", self.move_number)
                action2Return = self.m_buildBaseSubAgent.step(obs, True)
                self.move_number = self.m_buildBaseSubAgent.move_number
            
            elif self.current_action == ID_ACTION_BUILD_BASE:
                # print("build base action move num =", self.move_number)
                action2Return = self.m_buildBaseSubAgent.step(obs)
                self.move_number = self.m_buildBaseSubAgent.move_number

            elif self.current_action == ID_ACTION_TRAIN_ARMY:
                action2Return = self.m_trainArmySubAgent.step(obs)
                self.move_number = self.m_trainArmySubAgent.move_number
            
            elif self.current_action == ID_ACTION_ATTACK:
                action2Return = self.m_attackSubAgent.step(obs)
                self.move_number = self.m_attackSubAgent.move_number            

            if action2Return == DO_NOTHING_SC2_ACTION: 
                coord = self.m_buildBaseSubAgent.BuidingCheck()
                if coord[0] >= 0:
                    self.IsMultiSelect = True
                    action2Return = actions.FunctionCall(SC2_Actions.SELECT_POINT, [SC2_Params.SELECT_ALL, SwapPnt(coord)]) 

            return action2Return

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

    def FirstStep(self, obs):
        player_y, player_x = (obs.observation['minimap'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()

        if player_y.any() and player_y.mean() <= 31:
            self.base_top_left = True 
        else:
            self.base_top_left = False

        # action and state
        self.current_action = None
        self.move_number = 0
        self.previous_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')
        self.previous_scaled_state = np.zeros(STATE_SIZE, dtype=np.int32, order='C')

        self.m_buildBaseSubAgent.FirstStep(obs)
        self.m_trainArmySubAgent.FirstStep(obs)
        self.m_attackSubAgent.FirstStep(obs)

    def LastStep(self, obs):
        self.m_buildBaseSubAgent.LastStep(obs, obs.reward)
        self.m_trainArmySubAgent.LastStep(obs)
        self.m_attackSubAgent.LastStep(obs)

        self.EndRunQTable(obs)

        self.step_num = 0 

    def CreateState(self, obs):
        self.m_buildBaseSubAgent.CreateState(obs)
        self.m_trainArmySubAgent.CreateState(obs)
        self.m_attackSubAgent.CreateState(obs)

        self.ScaleCurrState()
    def Learn(self, obs):
        self.m_buildBaseSubAgent.Learn(obs)
        self.m_trainArmySubAgent.Learn(obs)
        self.m_attackSubAgent.Learn(obs)

        if self.current_action is not None:
            self.tables.learn(str(self.previous_scaled_state), self.current_action, 0, str(self.current_scaled_state))

    def EndRunQTable(self, obs):
        reward = obs.reward
        self.tables.learn(str(self.previous_scaled_state), self.current_action, reward, 'terminal')
        self.tables.end_run(reward)
       
    def ScaleCurrState(self):
        self.current_scaled_state[:] = self.current_state[:]

  
    def FindBuildingRightEdge(self, buildingType, point):
        buildingMat = self.unit_type == buildingType
        found = False
        x = point[SC2_Params.X_IDX]
        y = point[SC2_Params.Y_IDX]

        while not found:
            if x + 1 >= SC2_Params.SCREEN_SIZE:
                break 

            x += 1
            if not buildingMat[y][x]:
                if y + 1 < SC2_Params.SCREEN_SIZE and buildingMat[y + 1][x]:
                    y += 1
                elif y > 0 and buildingMat[y - 1][x]:
                    y -= 1
                else:
                    found = True

        return y,x
