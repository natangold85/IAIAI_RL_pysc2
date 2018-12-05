
from algo_qtable import QTableParamsExplorationDecay
from algo_dqn import DQN_PARAMS
from algo_a2c import A2C_PARAMS
from algo_a3c import A3C_PARAMS

# possible types of decision maker
QTABLE = 'qtable'
DQN = 'dqn'
DQN2L = 'dqn_2l'
DQN2L_EXPLORATION_CHANGE = 'dqn_2l_explorationChange'
A2C = 'A2C'
A3C = 'A3C'
DQN_EMBEDDING_LOCATIONS = 'dqn_Embedding'
HEURISTIC = 'heuristic' 

USER_PLAY = 'play'

# data for run type
ALGO_TYPE = "algo_type"
DECISION_MAKER_TYPE = "dm_type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "history"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'


AGENTS_PARAMS = {}

AGENTS_PARAMS["super_agent"] = {}
AGENTS_PARAMS["super_agent"]['numTrials2Save'] = 20
AGENTS_PARAMS["super_agent"]['numTrials4Cmp'] = 200

AGENTS_PARAMS["army_attack"] = {}
AGENTS_PARAMS["army_attack"]['numTrials2Save'] = 100
AGENTS_PARAMS["army_attack"]['numTrials4Cmp'] = 500

AGENTS_PARAMS["battle_mngr"] = {}
AGENTS_PARAMS["battle_mngr"]['numTrials2Save'] = 100
AGENTS_PARAMS["battle_mngr"]['numTrials4Cmp'] = 500

AGENTS_PARAMS["base_attack"] = {}
AGENTS_PARAMS["base_attack"]['numTrials2Save'] = 100
AGENTS_PARAMS["base_attack"]['numTrials4Cmp'] = 500

AGENTS_PARAMS["base_mngr"] = {}
AGENTS_PARAMS["base_mngr"]['numTrials2Save'] = 20
AGENTS_PARAMS["base_mngr"]['numTrials4Cmp'] = 200

AGENTS_PARAMS["build_base"] = {}
AGENTS_PARAMS["build_base"]['numTrials2Save'] = 20
AGENTS_PARAMS["build_base"]['numTrials4Cmp'] = 200

AGENTS_PARAMS["train_army"] = {}
AGENTS_PARAMS["train_army"]['numTrials2Save'] = 20
AGENTS_PARAMS["train_army"]['numTrials4Cmp'] = 200


def GetAgentParams(agentName):
    return AGENTS_PARAMS[agentName]

def GetRunType(agentName, configDict):
    runType = {}
    if configDict[agentName] == "none":
        return runType
    
    runType[HISTORY] = "history"
    runType[RESULTS] = "results"
    runType[DIRECTORY] = ""

    if configDict[agentName] == QTABLE:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "QLearningTable"
        runType[DIRECTORY] = agentName + "_qtable"
        runType[PARAMS] = QTableParamsExplorationDecay(0, 0, numTrials2Save=AGENTS_PARAMS[agentName]['numTrials2Save'])
        runType[DECISION_MAKER_NAME] = agentName + "_qtable"

    elif configDict[agentName] == DQN:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "DQN_WithTarget"
        runType[DIRECTORY] = agentName + "_dqn"
        runType[PARAMS] = DQN_PARAMS(0, 0, numTrials2Save=AGENTS_PARAMS[agentName]['numTrials2Save'], numTrials2CmpResults=AGENTS_PARAMS[agentName]['numTrials4Cmp'])
        runType[DECISION_MAKER_NAME] = agentName + "_dqn"

    elif configDict[agentName] == DQN2L:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "DQN_WithTarget"
        runType[DIRECTORY] = agentName + "_dqn"
        runType[PARAMS] = DQN_PARAMS(0, 0, layersNum=2, numTrials2Save=AGENTS_PARAMS[agentName]['numTrials2Save'], numTrials2CmpResults=AGENTS_PARAMS[agentName]['numTrials4Cmp'])
        runType[DECISION_MAKER_NAME] = agentName + "_dqn"

    elif configDict[agentName] == A2C:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "A2C"
        runType[DIRECTORY] = agentName + "_A2C"
        runType[PARAMS] = A2C_PARAMS(0, 0, numTrials2Save=AGENTS_PARAMS["army_attack"]['numTrials2Save'])
        runType[DECISION_MAKER_NAME] = agentName + "_A2C"

    elif configDict[agentName] == A3C:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerOnlineAsync"
        runType[ALGO_TYPE] = "A3C"
        runType[DIRECTORY] = agentName + "_A3C"
        runType[PARAMS] = A3C_PARAMS(0, 0, numTrials2Learn=1, numTrials2Save=AGENTS_PARAMS["army_attack"]['numTrials2Save'])
        runType[DECISION_MAKER_NAME] = agentName + "_A3C"

    elif configDict[agentName] == HEURISTIC:
        runType[HISTORY] = ""
        runType[RESULTS] = ""
        runType[DIRECTORY] = agentName + "_heuristic"

    elif configDict[agentName] == USER_PLAY:
        runType[HISTORY] = ""
        runType[RESULTS] = ""
        runType[DECISION_MAKER_TYPE] = "UserPlay"

    return runType







