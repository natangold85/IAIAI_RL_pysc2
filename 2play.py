
import numpy as np
import time
import datetime
import math

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions

STEP_DURATION = 0

DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])


class Play(base_agent.BaseAgent):
    def __init__(self):        
        super(Play, self).__init__()

        self.prevTime = None

    def step(self, obs):
        super(Play, self).step(obs)
        if obs.first():
            self.FirstStep(obs)
        elif obs.last():
             self.LastStep(obs)
             return DO_NOTHING_SC2_ACTION
        t = datetime.datetime.now()
        if self.prevTime != None:
            diff = t - self.prevTime
            #print("time(ms) = ", diff.seconds * 1000 + diff.microseconds / 1000)
        self.prevTime = t

        sc2Action = DO_NOTHING_SC2_ACTION
        time.sleep(STEP_DURATION)


        return sc2Action

    def FirstStep(self, obs):
        self.numStep = 0
             

    def LastStep(self, obs):
        if obs.reward > 0:
            print("won")
        elif obs.reward < 0:
            print("loss")
        else:
            print("tie")