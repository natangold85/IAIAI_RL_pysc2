#!/usr/bin/python3

# run example: python .\runSC2Agent.py --map=Simple64 --runDir=NaiveRunDiff2 --trainAgent=super --train=True
# kill all sc ps:  $ Taskkill /IM SC2_x64.exe /F

import logging
import traceback

import os
import threading
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

RUN = True

NUM_CRASHES = 0

NUM_CRASHES_2_RESTART = 5

RENDER = False
SCREEN_SIZE = SC2_Params.SCREEN_SIZE
MINIMAP_SIZE = SC2_Params.MINIMAP_SIZE

# general params
flags.DEFINE_string("do", "run", "what to  do: options =[run, check, copyNN]") 
flags.DEFINE_string("device", "gpu", "Which device to run nn on.")
flags.DEFINE_string("runDir", "none", "directory of the decision maker (should contain config file name config.txt)")

# for run:
flags.DEFINE_string("trainAgent", "none", "Which agent to train.")
flags.DEFINE_string("playAgent", "none", "Which agent to play.")
flags.DEFINE_string("train", "True", "open multiple threads for train.")
flags.DEFINE_string("map", "none", "Which map to run.")
flags.DEFINE_string("numSteps", "0", "num steps of map.")
flags.DEFINE_string("numGameThreads", "8", "num of game threads.")

# for check:
flags.DEFINE_string("checkAgent", "none", "Which agent to check.")
flags.DEFINE_string("fromDir", "none", "directory of the decision maker to copy from (should contain config file name config.txt)")
flags.DEFINE_string("stateIdx2Check", "0,1", "Which agent to check.")
flags.DEFINE_string("actions2Check", "0", "Which agent to check.")
flags.DEFINE_string("plot", "False", "Which agent to check.")

# for copy network
flags.DEFINE_string("copyAgent", "none", "Which agent to copy.")


nonRelevantRewardMap = ["BaseMngr"]
singlePlayerMaps = ["ArmyAttack5x5", "AttackBase", "AttackMngr"]


"""Script for starting all agents (a3c, very simple and slightly smarter).

This scripts is the starter for all agents, it has one command line parameter (--agent), that denotes which agent to run.
By default it runs the A3C agent.
"""
        
def run_thread(agent, display, players, numSteps):
    """Runs an agent thread."""

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


def start_agent():
    """Starts the pysc2 agent."""

    training = eval(flags.FLAGS.train)  
    if not training:
        parallel = 1
        show_render = True
    else:
        parallel = int(flags.FLAGS.numGameThreads)
        show_render = RENDER

    # tables
    dm_Types = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    dm_Types["directory"] = flags.FLAGS.runDir

    if flags.FLAGS.trainAgent == "none":
        trainList = ["super"]
    else:
        trainList = flags.FLAGS.trainAgent
        trainList = trainList.split(",")

    if flags.FLAGS.playAgent == "none":
        playList = trainList
    else:
        playList = flags.FLAGS.playAgent
        playList = playList.split(",")

    print("\n\n\nplay list =", playList)        
    print("train list =", trainList, "\n\n\n")
    
    isPlotThread = eval(flags.FLAGS.plot)
    
    isMultiThreaded = parallel > 1
    useMapRewards = flags.FLAGS.map not in nonRelevantRewardMap

    decisionMaker = None
    agents = []
    for i in range(parallel + isPlotThread):
        print("\n\n\n running thread #", i, "\n\n\n")
        agent = SuperAgent(decisionMaker=decisionMaker, isMultiThreaded=isMultiThreaded, dmTypes = dm_Types, playList=playList, trainList=trainList, useMapRewards=useMapRewards)
        
        if i == 0:
            decisionMaker = agent.GetDecisionMaker()

        agents.append(agent)

    threads = []
    show = show_render
    

    difficulty = int(dm_Types["difficulty"])
    players = [sc2_env.Agent(race=sc2_env.Race.terran)]
    if flags.FLAGS.map not in singlePlayerMaps:
        players.append(sc2_env.Bot(race=sc2_env.Race.terran, difficulty=difficulty))

    
    numSteps = int(flags.FLAGS.numSteps)

    idx = 0
    for i in range(parallel):
        thread_args = (agents[i], show, players, numSteps)
        t = threading.Thread(target=run_thread, args=thread_args, daemon=True)
        t.setName("GameThread_" + str(idx))

        threads.append(t)
        t.start()
        time.sleep(5)
        show = False
        idx += 1

    
    numTrials2Learn = [-1]
    if isPlotThread:
        dir2Save = "./" + dm_Types["directory"] + "/nnGraphs/"
        if not os.path.isdir(dir2Save):
            os.makedirs(dir2Save)
        thread_args = (agents[parallel], trainList[0], dir2Save, numTrials2Learn)
        t = threading.Thread(target=plot_thread, args=thread_args, daemon=True)
        t.setName("PlotThread")
        threads.append(t)
        t.start()


    runThreadAlive = True
    threading.current_thread().setName("TrainThread")
    

    while runThreadAlive:
        # train  when flag of training is on
        numTrials = decisionMaker.TrainAll()
        if numTrials >= 0:
            numTrials2Learn[0] = numTrials

        time.sleep(1)
        
        # if at least one thread is alive continue running
        runThreadAlive = False

        for t in threads:
            isAlive = t.isAlive()
            if isAlive:
                runThreadAlive = True
            else:
                t.join() 

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

def check_dqn():
    dm_Types = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    dm_Types["directory"] = flags.FLAGS.runDir
    
    checkList = flags.FLAGS.checkAgent.split(",")

    superAgent = SuperAgent(dmTypes = dm_Types)
    decisionMaker = superAgent.GetDecisionMaker()

    print("\n\nagent 2 check:", checkList, end='\n\n')
    np.set_printoptions(precision=2, suppress=True)
    for agentName in checkList:
        dm = decisionMaker.GetDecisionMakerByName(agentName)
        historyMngr = dm.historyMngr
        print(agentName, "hist size =", len(historyMngr.transitions["a"]))
        print(agentName, "old hist size", len(historyMngr.oldTransitions["a"]))
        print(agentName, "all history size", len(historyMngr.GetAllHist()["a"]))

        print(agentName, "maxVals =", historyMngr.transitions["maxStateVals"])
        print(agentName, "maxReward =", historyMngr.GetMaxReward(), "minReward =", historyMngr.GetMinReward())
        print("\n")
    
    numStates2Check = 100
    numEpochs = 10
    for sa in checkList:
        dm = decisionMaker.GetDecisionMakerByName(sa)

        states2Check = []
        for i in range(numStates2Check):
            s = dm.DrawStateFromHist()
            if len(s) > 0:
                states2Check.append(s)
        print(sa, ": current dqn num runs = ", dm.decisionMaker.NumRuns()," avg values =", avg_values(dm, states2Check))
        print(sa, ": target dqn num runs = ", dm.decisionMaker.NumRunsTarget()," avg values =", avg_values(dm, states2Check, True))
        
        if flags.FLAGS.runDir.find("Dflt") >= 0:
            print("dqn value =", dm.decisionMaker.ValueDqn(), "target value =", dm.decisionMaker.ValueTarget(), "heuristic values =", dm.decisionMaker.ValueDefault())
        else:
            print("dqn value =", dm.decisionMaker.ValueDqn(), "target value =", dm.decisionMaker.ValueTarget())

        print("\n\n")

    agent2Check = checkList[0]
    
    plotGraphs = eval(flags.FLAGS.plot)
    if plotGraphs:
        statesIdx = flags.FLAGS.stateIdx2Check.split(",")
        for i in range(len(statesIdx)):
            statesIdx[i] = int(statesIdx[i])

        actions2Check = flags.FLAGS.actions2Check.split(",")   
        for i in range(len(actions2Check)):
            actions2Check[i] = int(actions2Check[i])

        create_nnGraphs(superAgent, agent2Check, statesIdx=statesIdx, actions2Check=actions2Check)
        create_nnGraphs(superAgent, agent2Check, statesIdx=statesIdx, actions2Check=actions2Check, plotTarget=True, showGraphs=True)


def avg_values(dm, states, targetVals=False):
    vals = []
    for s in states:
        vals.append(dm.ActionsValues(s, targetVals))

    return np.average(vals, axis=0)


def copy_dqn():
    dmTypesSource = eval(open("./" + flags.FLAGS.fromDir + "/config.txt", "r+").read())
    dmTypesSource["directory"] = flags.FLAGS.fromDir

    superAgentSource = SuperAgent(dmTypes = dmTypesSource)
    decisionMakerSource = superAgentSource.GetDecisionMaker()    

    dmTypesTarget = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    dmTypesTarget["directory"] = flags.FLAGS.runDir
    
    superAgentTarget = SuperAgent(dmTypes = dmTypesTarget)
    decisionMakerTarget = superAgentTarget.GetDecisionMaker()    
    
    copyList = flags.FLAGS.copyAgent
    copyList = copyList.split(",")

    for agent in copyList:
        currDmSource = decisionMakerSource.GetDecisionMakerByName(agent)
        currDmTarget = decisionMakerTarget.GetDecisionMakerByName(agent)

        if currDmSource != None and currDmTarget != None:
            allVarsSource, _ = currDmSource.decisionMaker.GetAllNNVars()
            currDmTarget.decisionMaker.AssignAllNNVars(allVarsSource)
            currDmTarget.decisionMaker.SaveDQN()
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

    if flags.FLAGS.do == "run":
        start_agent()
    elif flags.FLAGS.do == "check":
        check_dqn()
    elif flags.FLAGS.do == "copyNN":
        copy_dqn()


if __name__ == '__main__':
    print('Starting...')
    app.run(main)
