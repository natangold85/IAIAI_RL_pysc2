#!/usr/bin/python3

# run example: python .\runSC2Agent.py --map=Simple64 --runDir=NaiveRunDiff2 --trainAgent=super --train=True
# kill all sc ps:  $ Taskkill /IM SC2_x64.exe /F

import logging
import traceback

import os
import sys
import threading
from multiprocessing import Pool                                                
import time
import tensorflow as tf
import collections
from absl import app
from absl import flags
import math
import numpy as np

from pysc2.env import run_loop
from pysc2.env import sc2_env

# all independent agents available
from agent_super import SuperAgent

from utils import SC2_Params
from utils_plot import create_nnGraphs

from paramsCalibration import GeneticProgrammingGeneration
from paramsCalibration import ReadGPFitness
from paramsCalibration import TrainSingleGP

# agent possible to run calibration
from agent_army_attack import CreateDecisionMakerArmyAttack
from agent_army_attack import GetRunTypeArmyAttack

RUN = True

MAX_NUM_OF_TEST_SIMULTANEOUSLY = 4
NUM_CRASHES = 0

NUM_CRASHES_2_RESTART = 5

RENDER = False
SCREEN_SIZE = SC2_Params.SCREEN_SIZE
MINIMAP_SIZE = SC2_Params.MINIMAP_SIZE

# general params
flags.DEFINE_string("act", "run", "what to  act: options =[run, check, copyNN, gp]") 
flags.DEFINE_string("device", "gpu", "Which device to run nn on.")
flags.DEFINE_string("runDir", "none", "directory of the decision maker (should contain config file name config.txt)")

# params for genetic programming
flags.DEFINE_string("gpAct", "trainPopulation", "train population or test single individual")
flags.DEFINE_string("populationSize", "20", "population size")
flags.DEFINE_string("populationInstances", "2", "num instances for each individual")
flags.DEFINE_string("numGenerations", "20", "num generations to run")
flags.DEFINE_string("populationIdx", "none", "which idx of population to test or train")
flags.DEFINE_string("populationInstanceIdx", "none", "which instance of individual to test or train")

# for run:
flags.DEFINE_string("testAgent", "none", "Which agent to test.")
flags.DEFINE_string("trainAgent", "none", "Which agent to train.")
flags.DEFINE_string("playAgent", "none", "Which agent to play.")
flags.DEFINE_string("map", "none", "Which map to run.")
flags.DEFINE_string("numSteps", "0", "num steps of map.")
flags.DEFINE_string("numGameThreads", "1", "num of game threads.")
flags.DEFINE_string("numEpisodes", "none", "num of episodes agent to run.")

# for check:
flags.DEFINE_string("checkAgent", "none", "Which agent to check.")
flags.DEFINE_string("fromDir", "none", "directory of the decision maker to copy from (should contain config file name config.txt)")
flags.DEFINE_string("stateIdx2Check", "0,1", "Which agent to check.")
flags.DEFINE_string("actions2Check", "0", "Which agent to check.")
flags.DEFINE_string("plot", "False", "Which agent to check.")

flags.DEFINE_string("resetModel", "False", "if to reset data(dm params, history and results)")

# for copy network
flags.DEFINE_string("copyAgent", "none", "Which agent to copy.")

nonRelevantRewardMap = ["BaseMngr"]
singlePlayerMaps = ["ArmyAttack5x5", "AttackBase", "AttackMngr"]

def start_agent(configDict=None, copy2Run=None, threadName="Thread"):
    """Starts the pysc2 agent."""
    
    # if configDict is not argument read it from file
    if configDict == None:
        configDictOrg = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
        configDict = configDictOrg.copy()
        configDict["directory"] = flags.FLAGS.runDir

    # parse list of agent participating in session
    trainList = flags.FLAGS.trainAgent
    trainList = trainList.split(",")

    testList = flags.FLAGS.testAgent
    testList = testList.split(",")
    
    training = trainList != ["none"]

    if flags.FLAGS.playAgent == "none":
        playList = trainList if trainList != ["none"] else testList
    else:
        playList = flags.FLAGS.playAgent
        playList = playList.split(",")

    print("\n\n\nplay list =", playList)        
    print("train list =", trainList)
    print("test list =", testList, "\n\n\n")

    # parse params from flags and configDict

    if flags.FLAGS.numEpisodes == "none":
        numEpisodes = None
    else:
        numEpisodes = int(flags.FLAGS.numEpisodes)

    if "numRun" in configDict.keys():
        numRun = configDict["numRun"]
    else:
        numRun = None

    if "numGameThreads" in configDict.keys():
        numDmThreads = configDict["numGameThreads"]
    else:
        numDmThreads = int(flags.FLAGS.numGameThreads)

    numGameThreads = numDmThreads if training else 1

    if "sharedDM" in configDict.keys():
        sharedDM = configDict["sharedDM"]
    else:
        sharedDM = True


    
    isPlotThread = eval(flags.FLAGS.plot)
    
    isMultiThreaded = numGameThreads > 1
    useMapRewards = flags.FLAGS.map not in nonRelevantRewardMap

    difficulty = int(configDict["difficulty"])
    players = [sc2_env.Agent(race=sc2_env.Race.terran)]
    if flags.FLAGS.map not in singlePlayerMaps:
        players.append(sc2_env.Bot(race=sc2_env.Race.terran, difficulty=difficulty))

    allDecisionMakers = []
    decisionMaker = None
    agents = []
    for i in range(numDmThreads + isPlotThread):
        if copy2Run != None:
            dmCopy = copy2Run
        else:
            if not sharedDM:
                dmCopy = i
            else:
                dmCopy = numRun

        print("\n\n\n init decision maker instance #", i, "\n\n\n")
        agent = SuperAgent(decisionMaker=decisionMaker, isMultiThreaded=isMultiThreaded, configDict=configDict, playList=playList, trainList=trainList, testList=testList, useMapRewards=useMapRewards, dmCopy=dmCopy)
        
        if not sharedDM:
            allDecisionMakers.append(agent.GetDecisionMaker())
        elif i == 0:
            decisionMaker = agent.GetDecisionMaker()
            allDecisionMakers.append(decisionMaker)

        agents.append(agent)

    with tf.Session() as sess:
        # create savers
        resetModel = eval(flags.FLAGS.resetModel)
        for dm in allDecisionMakers:
            dm.InitModel(sess, resetModel)

        threads = []        

        numSteps = int(flags.FLAGS.numSteps)

        idx = 0
        for i in range(numGameThreads):
            print("\n\n\n init game thread #", i, "\n\n\n")

            thread_args = (agents[i], sess, RENDER, players, numSteps)
            t = threading.Thread(target=run_thread, args=thread_args, daemon=True)
            t.setName( "Game" + threadName + "_" + str(idx))

            threads.append(t)
            t.start()
            time.sleep(5)
            idx += 1


        numTrials2Learn = [-1]
        if isPlotThread:
            dir2Save = "./" + configDict["directory"] + "/nnGraphs/"
            if not os.path.isdir(dir2Save):
                os.makedirs(dir2Save)
            thread_args = (agents[numGameThreads], trainList[0], dir2Save, numTrials2Learn)
            t = threading.Thread(target=plot_thread, args=thread_args, daemon=True)
            t.setName("PlotThread")
            threads.append(t)
            t.start()


        contRun = True
        threading.current_thread().setName("Train" + threadName)
        

        while contRun:
            # train  when flag of training is on
            for dm in allDecisionMakers:
                numTrials = dm.TrainAll()
                if numTrials >= 0:
                    numTrials2Learn[0] = numTrials

            time.sleep(0.5)
            
            # if at least one thread is alive continue running
            contRun = False

            for t in threads:
                isAlive = t.isAlive()
                if isAlive:
                    contRun = True
                else:
                    t.join() 

            if numEpisodes != None:
                minRuns = numEpisodes + 1
                for dm in allDecisionMakers:
                    if trainList[0] != "none":
                        dmAgent = dm.GetDecisionMakerByName(trainList[0])
                        minRuns = min(dmAgent.NumRuns(), minRuns)
                    else:
                        dmAgent = dm.GetDecisionMakerByName(testList[0])
                        minRuns = min(dmAgent.NumTestRuns(), minRuns)
                
                if minRuns > numEpisodes:
                    if "numRun" in configDict:
                        print("\n\nending run #", configDict["numRun"], "!!!\n\n") 
                        configDictOrg["numRun"] += 1
                        open("./" + flags.FLAGS.runDir + "/config.txt", "w+").write(str(configDictOrg))
                    if testList[0] != "none":
                        dm.GetDecisionMakerByName(testList[0]).Save()

                    contRun = False

def run_script(gpActArg, populationIdx, instanceIdx):     
    cmd = ' '.join(sys.argv)
    cmd.replace(".\\", "")                                                        
    os.system('python {}'.format(cmd + " --gpAct=" + gpActArg + " --populationIdx=" + str(populationIdx) + " --populationInstanceIdx=" + str(instanceIdx)))    

def run_gp_threads(populationSize, numInstances, gpActArg):
    numThreadsRunning = 10
    numOfTerminals2KillGame = 10

    threadsB4Run = []  
    for idx in range(populationSize):
        for instance in range(numInstances):
            t = threading.Thread(target=run_script, args=(gpActArg, idx, instance))
            threadsB4Run.append(t)
    
    numTerminal = 0
    runningThreads = []

    while len(threadsB4Run) > 0 or len(runningThreads) > 0:
        if len(runningThreads) < numThreadsRunning and len(threadsB4Run) > 0:
            t = threadsB4Run.pop()
            t.start()
            runningThreads.append(t)

        for t in runningThreads:
            if not t.isAlive():
                numTerminal += 1
                runningThreads.remove(t)

        if numTerminal >= numOfTerminals2KillGame and gpActArg == "testSingle":
            os.system('"Taskkill /IM SC2_x64.exe /F"')
            numTerminal = 0
    
    if gpActArg == "testSingle":
        os.system('"Taskkill /IM SC2_x64.exe /F"')     
            
def gp_train(): 
    print("\n\n\ntrain gp\n\n\n")

    numGenerations = int(flags.FLAGS.numGenerations)
    populationSize = int(flags.FLAGS.populationSize)
    numPopulationInstances = int(flags.FLAGS.populationInstances)
    
    configDict = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict["directory"] = flags.FLAGS.runDir

    trainAgent = flags.FLAGS.testAgent

    if trainAgent == "army_attack":
        runType = GetRunTypeArmyAttack(configDict)


    for gen in range(numGenerations):

        print("\n\n\ncreate population generation #", gen, "\n\n\n")
        GeneticProgrammingGeneration(populationSize, numPopulationInstances, configDict, runType)

        print("\n\n\ntrain population generation #", gen, "\n\n\n")
        run_gp_threads(populationSize, numPopulationInstances, "trainSingle")

        print("\n\n\ntest population generation #", gen, "\n\n\n")
        run_gp_threads(populationSize, numPopulationInstances, "testSingle")

        ReadGPFitness(configDict, runType)


def getGP_Params(population, params2Calibrate, populationIdx, instanceIdx):
    individual2Run = population[populationIdx]

    dirsCopy2Run = individual2Run[0][instanceIdx]
    paramDict = {}
    for i in range(len(params2Calibrate)):
        paramDict[params2Calibrate[i]] = individual2Run[1][i]    
    
    return dirsCopy2Run, paramDict

def gp_train_single():  
    populationIdx = int(flags.FLAGS.populationIdx)
    instance2Train  = int(flags.FLAGS.populationInstanceIdx)

    configDict = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict["directory"] = flags.FLAGS.runDir

    gpPopulation = eval(open("./" + flags.FLAGS.runDir + "/gp_population.txt", "r+").read())

    params2Calibrate = configDict["params2Calibrate"]

    dirsCopy2Run, paramDict = getGP_Params(gpPopulation["population"], params2Calibrate, populationIdx, instance2Train)
    
    print("\n\n\ntrain single", populationIdx, "instance =", instance2Train)
    print(populationIdx, ": dir copy =", dirsCopy2Run, "params dict =", paramDict, "\n\n\n")

    configDict["hyperParams"] = paramDict
    
    trainAgent = flags.FLAGS.testAgent
    if trainAgent == "army_attack":
        dmInitFunc = CreateDecisionMakerArmyAttack
        runType = GetRunTypeArmyAttack(configDict)

    threading.current_thread().setName("TrainSingle_" + str(populationIdx) + "_" + str(instance2Train))

    TrainSingleGP(configDict, runType, dmInitFunc, dirsCopy2Run)

def gp_test_single(): 
    populationIdx = int(flags.FLAGS.populationIdx)
    instance2Train  = int(flags.FLAGS.populationInstanceIdx)


    configDictOrg = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict = configDictOrg.copy()
    configDict["directory"] = flags.FLAGS.runDir

    gpPopulation = eval(open("./" + flags.FLAGS.runDir + "/gp_population.txt", "r+").read())
    params2Calibrate = configDict["params2Calibrate"]

    dirsCopy2Run, paramDict = getGP_Params(gpPopulation["population"], params2Calibrate, populationIdx, instance2Train) 
    
    print("test single", populationIdx, "instance =", instance2Train, "dircopyNum =", dirsCopy2Run, "\n\n\n") 
    
    configDict["hyperParams"] = paramDict
    configDict["numGeneration"] = gpPopulation["numGeneration"]

    start_agent(configDict, dirsCopy2Run, threadName="pop#" + str(populationIdx) + "_" + str(instance2Train))
    
def run_thread(agent, sess, display, players, numSteps):
    """Runs an agent thread."""

    with sess.as_default(), sess.graph.as_default():

        while RUN:
            try:

                agent_interface_format=sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(screen=SCREEN_SIZE,minimap=MINIMAP_SIZE))

                with sc2_env.SC2Env(map_name=flags.FLAGS.map,
                                    players=players,
                                    game_steps_per_episode=numSteps,
                                    agent_interface_format=agent_interface_format,
                                    visualize=display) as env:
                    run_loop.run_loop([agent], env)

            
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                logging.error(traceback.format_exc())
            
            # remove crahsed terminal history
            # agent.RemoveNonTerminalHistory()

            global NUM_CRASHES
            NUM_CRASHES += 1

def plot_thread(agent, agent2Train, dir2Save, numTrials2Learn):
    statesIdx = flags.FLAGS.stateIdx2Check.split(",")
    for i in range(len(statesIdx)):
        statesIdx[i] = int(statesIdx[i])

    actions2Check = flags.FLAGS.actions2Check.split(",")   
    for i in range(len(actions2Check)):
        actions2Check[i] = int(actions2Check[i])

    while True:
        if numTrials2Learn[0] >= 0:
            numTrials = numTrials2Learn[0]
            numTrials2Learn[0] = -1
            create_nnGraphs(agent, agent2Train, statesIdx=statesIdx, actions2Check=actions2Check, numTrials=numTrials, saveGraphs=True, dir2Save = dir2Save)
        time.sleep(1)

def check_model():
    configDict = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict["directory"] = flags.FLAGS.runDir

    checkList = flags.FLAGS.checkAgent.split(",")
    sharedDM = True if "sharedDM" not in configDict.keys() else configDict["sharedDM"]
    dmCopy = None if sharedDM else 0
    superAgent = SuperAgent(configDict=configDict, dmCopy=dmCopy)

    print("\n\nagent 2 check:", checkList, end='\n\n')
    with tf.Session() as sess:
        # create savers
        decisionMaker = superAgent.GetDecisionMaker()
        decisionMaker.InitModel(sess, resetModel=False)    

        for agentName in checkList:
            agent = superAgent.GetAgentByName(agentName)
            plotGraphs = eval(flags.FLAGS.plot)

            statesIdx = list(map(int, flags.FLAGS.stateIdx2Check.split(",")))
            actions2Check = list(map(int, flags.FLAGS.actions2Check.split(",")))

            withDfltVals = flags.FLAGS.runDir.find("Dflt") >= 0

            agent = superAgent.GetAgentByName(agentName)
            decisionMaker = agent.decisionMaker
            decisionMaker.CheckModel(agent, plotGraphs=plotGraphs, withDfltModel=withDfltVals, statesIdx2Check=statesIdx, actions2Check=actions2Check)

def copy_dqn():
    configDictSource = eval(open("./" + flags.FLAGS.fromDir + "/config.txt", "r+").read())
    configDictSource["directory"] = flags.FLAGS.fromDir

    superAgentSource = SuperAgent(configDict = configDictSource)
    decisionMakerSource = superAgentSource.GetDecisionMaker()    

    configDictTarget = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDictTarget["directory"] = flags.FLAGS.runDir
    
    superAgentTarget = SuperAgent(configDict = configDictTarget)
    decisionMakerTarget = superAgentTarget.GetDecisionMaker()    
    
    copyList = flags.FLAGS.copyAgent
    copyList = copyList.split(",")

    for agent in copyList:
        currDmSource = decisionMakerSource.GetDecisionMakerByName(agent)
        currDmTarget = decisionMakerTarget.GetDecisionMakerByName(agent)

        if currDmSource != None and currDmTarget != None:
            allVarsSource, _ = currDmSource.decisionMaker.GetAllNNVars()
            currDmTarget.decisionMaker.AssignAllNNVars(allVarsSource)
            currDmTarget.decisionMaker.Save()
        else:
            print("Error in agent = ", agent)
            print("source =", type(currDmSource))
            print("target =", type(currDmTarget))

       


def main(argv):
    """Main function.

    This function check which agent was specified as command line parameter and launches it.

    :param argv: empty
    :return:
    """

    if flags.FLAGS.device == "cpu":
        # run from cpu
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    ## create history of run commands text file

    modelDir = "./" + flags.FLAGS.runDir
    histCmdFName = modelDir + "/cmdHistory.txt"
    if not os.path.isdir(modelDir):
        os.makedirs(modelDir)

    if flags.FLAGS.resetModel == "True":
        open(histCmdFName, "w+").write(str(sys.argv))
    else:
        open(histCmdFName, "a+").write("\n\n" + str(sys.argv))

    ## call the act function
    if flags.FLAGS.act == "run":
        start_agent()
    elif flags.FLAGS.act == "check":
        check_model()
    elif flags.FLAGS.act == "copyNN":
        copy_dqn()
    elif flags.FLAGS.act == "gp":
        if flags.FLAGS.gpAct == "trainPopulation":
            gp_train()
        elif flags.FLAGS.gpAct == "trainSingle":
            gp_train_single()
        elif flags.FLAGS.gpAct == "testSingle":
            gp_test_single()



if __name__ == '__main__':
    print('Starting...')
    app.run(main)
