
import time
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from utils import TerranUnit
from utils import SC2_Params
from utils import SC2_Actions
from utils import QLearningTable

from utils import SwapPnt
from utils import PrintScreen
from utils import PrintMiniMap
from utils import PrintSpecificMat
from utils import FindMiddle

STEP_DURATION = 0.1

def GetSelfLocation(obs):
    self_y, self_x = (obs.observation['screen'][SC2_Params.PLAYER_RELATIVE] == SC2_Params.PLAYER_SELF).nonzero()
    if len(self_y) > 0:
        return FindMiddle(self_y, self_x)
    else:
        return [-1,-1]

class Play(base_agent.BaseAgent):
    def __init__(self):
        super(Play, self).__init__()

        self.queue = None

    def step(self, obs):
        super(Play, self).step(obs)
        
        if obs.last():
            print("\n\nend of trial")    
        time.sleep(STEP_DURATION)
        PrintScreen(obs)
 

        return actions.FunctionCall(SC2_Actions.NO_OP, [])