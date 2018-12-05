# import states
from agent_super import SUPER_STATE
from agent_build_base import BUILD_STATE
from agent_train_army import TRAIN_STATE
from agent_base_mngr import BASE_STATE
from agent_army_attack import ArmyAttackState
from agent_battle_mngr import BattleMngrState
from agent_base_attack import BaseAttackState

# import actions
from agent_super import SUPER_ACTIONS
from agent_base_mngr import BASE_ACTIONS
from agent_army_attack import ArmyAttackActions
from agent_build_base import BUILD_ACTIONS
from agent_train_army import TRAIN_ACTIONS
from agent_battle_mngr import BattleMngrActions
from agent_base_attack import BaseAttackActions

STATES = {}
ACTIONS = {}

def StateSize2Agent(agentName):
    return STATES[agentName].SIZE

def NumActions2Agent(agentName):
    return ACTIONS[agentName].SIZE

STATES["super_agent"] = SUPER_STATE()
STATES["base_mngr"] = BASE_STATE()
STATES["build_base"] = BUILD_STATE()
STATES["train_army"] = TRAIN_STATE()
STATES["army_attack"] = ArmyAttackState()
STATES["battle_mngr"] = BattleMngrState()
STATES["base_attack"] = BaseAttackState()

ACTIONS["super_agent"] = SUPER_ACTIONS()
ACTIONS["base_mngr"] = BASE_ACTIONS()
ACTIONS["build_base"] = BUILD_ACTIONS()
ACTIONS["train_army"] = TRAIN_ACTIONS()
ACTIONS["army_attack"] = ArmyAttackActions()
ACTIONS["battle_mngr"] = BattleMngrActions()
ACTIONS["base_attack"] = BaseAttackActions()

