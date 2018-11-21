
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

STEP_DURATION = 0.2
DO_NOTHING_SC2_ACTION = actions.FunctionCall(SC2_Actions.NO_OP, [])

# $ python -m pysc2.bin.agent --agent 2play.Play --map BaseMngr --max_agent_steps=0 --agent_race=terran

class Play(base_agent.BaseAgent):
    def __init__(self):        
        super(Play, self).__init__()
        self.maxSteps = 0
        self.prevTime = None

    def step(self, obs):
        super(Play, self).step(obs)
        sc2Action = DO_NOTHING_SC2_ACTION
        
        time.sleep(STEP_DURATION)
    
        return DO_NOTHING_SC2_ACTION


    def FirstStep(self, obs):
        pass

    def LastStep(self, obs):
        if obs.reward > 0:
            print("won")
        elif obs.reward < 0:
            print("loss")
        else:
            print("tie")