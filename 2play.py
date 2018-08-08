
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

from utils import PrintScreen

STEP_DURATION = 0.1
DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

#python -m pysc2.bin.agent --agent 2play.Play --map mapName --max_agent_steps=0 --agent_race=T --bot_race=T

class Play(base_agent.BaseAgent):
    def __init__(self):        
        super(Play, self).__init__()

        self.prevTime = None

    def step(self, obs):
        super(Play, self).step(obs)

        sc2Action = DO_NOTHING_SC2_ACTION
        
        if obs.first():
            self.FirstStep(obs)
            sc2Action = actions.FunctionCall(SC2_Actions.SELECT_ARMY, [SC2_Params.NOT_QUEUED])
        elif self.numStep == 1:
            sc2Action = actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_SET, [4]])

        elif self.numStep % 10 == 0:
            sc2Action = actions.FunctionCall(SC2_Actions.SELECT_CONTROL_GROUP, [SC2_Params.CONTROL_GROUP_RECALL, [4]])

        time.sleep(STEP_DURATION)

        self.numStep += 1
        return DO_NOTHING_SC2_ACTION


    def FirstStep(self, obs):
        self.numStep = 0
             

    def LastStep(self, obs):
        if obs.reward > 0:
            print("won")
        elif obs.reward < 0:
            print("loss")
        else:
            print("tie")