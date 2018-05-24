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

from build_base import BuildBaseSubAgent
from train_army import TrainArmySubAgent

Q_TABLE_BUILDBASE_FILE = 'buildbase_q_table'
T_TABLE_BUILDBASE_FILE = 'buildbase_t_table'

STEP_DURATION = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

UNIT_2_REWARD = {}
UNIT_2_REWARD[SC2_Actions.TRAIN_MARINE] = 1
UNIT_2_REWARD[SC2_Actions.TRAIN_REAPER] = 2
UNIT_2_REWARD[SC2_Actions.TRAIN_HELLION] = 4
UNIT_2_REWARD[SC2_Actions.TRAIN_SIEGE_TANK] = 8

NORMALIZATION = 300

class BuildBase(base_agent.BaseAgent):
    def __init__(self):
        super(BuildBase, self).__init__()
        
        # sub agents
        self.m_buildBaseSubAgent = BuildBaseSubAgent(Q_TABLE_BUILDBASE_FILE, T_TABLE_BUILDBASE_FILE, True)
        self.m_trainArmySubAgent = TrainArmySubAgent()

        self.lastTrain = None
        # model params 
        self.move_number = 0
        self.step_num = 0
        self.IsMultiSelect = False

        self.accumulatedReward = 0
    def step(self, obs):
        super(BuildBase, self).step(obs)
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

            action2Return = self.m_buildBaseSubAgent.stepForSolo(obs)

            if action2Return == "train_army":
                action2Return = self.m_trainArmySubAgent.step(obs)
                self.move_number = self.m_trainArmySubAgent.move_number
            else:
                self.move_number = self.m_buildBaseSubAgent.move_number            

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
        # action and state
        self.current_action = None
        self.move_number = 0
        self.accumulatedReward = 0

        self.m_buildBaseSubAgent.FirstStep(obs)
        self.m_trainArmySubAgent.FirstStep(obs)

    def LastStep(self, obs):
        reward = self.accumulatedReward - 1
        self.m_buildBaseSubAgent.LastStep(obs, reward)
        self.m_trainArmySubAgent.LastStep(obs)

        self.step_num = 0 

    def CreateState(self, obs):
        self.m_buildBaseSubAgent.CreateState(obs)
        self.m_trainArmySubAgent.CreateState(obs)

    def Learn(self, obs):
        lastTrain = self.m_trainArmySubAgent.GetLastUnitTrain()
        
        if lastTrain != None:
            reward = UNIT_2_REWARD[lastTrain] / NORMALIZATION
        else:
            reward = 0

        self.m_buildBaseSubAgent.Learn(obs, reward)
        self.accumulatedReward += reward