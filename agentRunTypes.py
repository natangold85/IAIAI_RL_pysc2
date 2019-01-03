
from algo_qtable import QTableParamsExplorationDecay
from algo_dqn import DQN_PARAMS
from algo_a2c import A2C_PARAMS
from algo_a3c import A3C_PARAMS

# possible types of decision maker
QTABLE = 'qtable'
DQN = 'dqn'
A2C = 'A2C'
A3C = 'A3C'
HEURISTIC = 'heuristic' 
USER_PLAY = 'play'

# layers num options
TWO_LAYERS = '2l'

# A2C options
ACCUMULATE_HISTORY = 'Exp'
ADJUSTED_MODEL_2_ACCUMULATION = 'Adjusted'

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
    runArg = configDict[agentName]
    if runArg == "none":
        return runType
    
    runType[HISTORY] = "history"
    runType[RESULTS] = "results"
    runType[DIRECTORY] = ""
    
    learningRate = None
    if "learning_rate" in configDict.keys():
        learningRate = configDict["learning_rate"]
    elif "learning_ratePower" in configDict.keys():
        learningRate = 10 ** configDict["learning_ratePower"]

    if runArg == QTABLE:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "QLearningTable"
        runType[DIRECTORY] = agentName + "_qtable"
        runType[PARAMS] = QTableParamsExplorationDecay(0, 0, numTrials2Save=AGENTS_PARAMS[agentName]['numTrials2Save'])
        runType[PARAMS].learning_rate = runType[PARAMS].learning_rate if learningRate == None else learningRate
        runType[DECISION_MAKER_NAME] = agentName + "_qtable"

    elif runArg == HEURISTIC:
        runType[HISTORY] = ""
        runType[DIRECTORY] = agentName + "_heuristic"

    elif runArg == USER_PLAY:
        runType[HISTORY] = ""
        runType[RESULTS] = ""
        runType[DECISION_MAKER_TYPE] = "UserPlay"

    else:
        # neural nets model: 
        layersNum = 2 if runArg.find(TWO_LAYERS) >= 0 else 1
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        numTrials2Save = numTrials2Save=AGENTS_PARAMS[agentName]['numTrials2Save']

        if runArg.find(DQN) >= 0:
            runType[ALGO_TYPE] = "DQN_WithTarget"
            typeStr = DQN
            runType[PARAMS] = DQN_PARAMS(0, 0, layersNum=layersNum, numTrials2Save=numTrials2Save, numTrials2CmpResults=AGENTS_PARAMS[agentName]['numTrials4Cmp'])
            runType[PARAMS].learning_rate = runType[PARAMS].learning_rate if learningRate == None else learningRate

        elif runArg.find(A2C) >= 0:
            typeStr = A2C
            runType[ALGO_TYPE] = "A2C"
            accumulateHistory = True if runArg.find(ACCUMULATE_HISTORY) >= 0 else False

            runType[PARAMS] = A2C_PARAMS(0, 0, accumulateHistory=accumulateHistory, layersNum=2, numTrials2Save=numTrials2Save)
            runType[PARAMS].learning_rateActor = runType[PARAMS].learning_rateActor if learningRate == None else learningRate
            runType[PARAMS].learning_rateCritic = runType[PARAMS].learning_rateCritic if learningRate == None else learningRate

        elif runArg.find(A3C):
            typeStr = A3C
            runType[DECISION_MAKER_TYPE] = "DecisionMakerOnlineAsync"
            runType[ALGO_TYPE] = "A3C"
            runType[PARAMS] = A3C_PARAMS(0, 0, numTrials2Learn=1, numTrials2Save=numTrials2Save)
            runType[PARAMS].learning_rate = runType[PARAMS].learning_rate if learningRate == None else learningRate
            runType[DECISION_MAKER_NAME] = agentName + "_A2C"

        runType[DIRECTORY] = agentName + "_" + typeStr
        runType[DECISION_MAKER_NAME] = agentName + "_" + typeStr

    return runType







