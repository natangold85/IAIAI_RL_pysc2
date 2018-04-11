
import time
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id

STEP_DURATION = 0.1
class Play(base_agent.BaseAgent):
    def __init__(self):
        super(Play, self).__init__()

        self.queue = None

    def step(self, obs):
        super(Play, self).step(obs)
        
        time.sleep(STEP_DURATION)
        print (obs.observation['player'][4])

        return actions.FunctionCall(_NO_OP, [])