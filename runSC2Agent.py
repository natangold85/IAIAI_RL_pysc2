#!/usr/bin/python3

# run example: python .\runSC2Agent.py --map=Simple64 --configFile=NaiveRunDiff2.txt --trainAgent=super --train=True
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
import numpy as np

from pysc2.env import run_loop
from pysc2.env import sc2_env

# all independent agents available
from agent_super import SuperAgent

from utils import SC2_Params

RUN = True

NUM_CRASHES = 0

NUM_CRASHES_2_RESTART = 5

RENDER = False
SCREEN_SIZE = SC2_Params.SCREEN_SIZE
MINIMAP_SIZE = SC2_Params.MINIMAP_SIZE

flags.DEFINE_string("trainAgent", "none", "Which agent to train.")
flags.DEFINE_string("playAgent", "none", "Which agent to play.")
flags.DEFINE_string("configFile", "none", "config file that builds heirarchi for decision maker (should contain a defenition of a dictionary)")
flags.DEFINE_string("train", "True", "open multiple threads for train.")
flags.DEFINE_string("map", "none", "Which map to run.")
flags.DEFINE_string("numSteps", "0", "num steps of map.")
flags.DEFINE_string("device", "gpu", "Which device to run nn on.")
flags.DEFINE_string("numGameThreads", "8", "num of game threads.")
flags.DEFINE_string("checkDqnValues", "False", "num of game threads.")

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
    dm_Types = eval(open(flags.FLAGS.configFile, "r+").read())
    dm_Types["directory"] = flags.FLAGS.configFile.replace(".txt", "")

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
    isMultiThreaded = parallel > 1

    useMapRewards = flags.FLAGS.map not in nonRelevantRewardMap

    decisionMaker = None
    agents = []
    for i in range(parallel):
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
    for agent in agents:
        thread_args = (agent, show, players, numSteps)
        t = threading.Thread(target=run_thread, args=thread_args)
        t.setName("GameThread_" + str(idx))

        threads.append(t)
        t.start()
        time.sleep(5)
        show = False
        idx += 1


    runThreadAlive = True
    threading.current_thread().setName("TrainThread")
    while runThreadAlive:
        # train  when flag of training is on
        decisionMaker.TrainAll()
        time.sleep(1)
        
        # if at least one thread is alive continue running
        runThreadAlive = False
        for t in threads:
            
            isAlive = t.isAlive()
            if isAlive:
                runThreadAlive = True
            else:
                t.join() 

        global NUM_CRASHES
        if NUM_CRASHES > NUM_CRASHES_2_RESTART or not RUN:
            #os.system('powershell.exe [Taskkill /IM SC2_x64.exe /F]')
            print("\n\ninit all sc2 games\n\n")
            NUM_CRASHES = 0


def check_dqn():
    dm_Types = eval(open(flags.FLAGS.configFile, "r+").read())
    dm_Types["directory"] = flags.FLAGS.configFile.replace(".txt", "")
    checkList = flags.FLAGS.playAgent
    checkList = checkList.split(",")


    superAgent = SuperAgent(dmTypes = dm_Types, playList=["super"], trainList=["super"])
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
        print("\n")
    
    numStates2Check = 100
    numEpochs = 10
    for sa in checkList:
        dm = decisionMaker.GetDecisionMakerByName(sa)
        s, a, r, s_, terminal = dm.historyMngr.GetHistory()

        states2Check = []
        for i in range(numStates2Check):
            states2Check.append(dm.DrawStateFromHist())
        print(sa, ": current dqn num runs = ", dm.decisionMaker.NumRuns()," avg values =", avg_values(dm, states2Check))
        print(sa, ": target dqn num runs = ", dm.decisionMaker.NumRunsTarget()," avg values =", avg_values(dm, states2Check, True))

        print("\n\n")


        # dm.decisionMaker.Reset()

        # print("epoch -1 : values =", avg_values(dm, states2Check))
        # for i in range(numEpochs):
        #     dm.decisionMaker.learn(s,a,r,s_, terminal)
        #     print("epoch", i,": values =", avg_values(dm, states2Check))
            
def avg_values(decisionMaker, states2Check, targetVals = False):
    vals = np.zeros(decisionMaker.params.numActions, float)
    for s in states2Check:
        vals += decisionMaker.ActionValuesVec(s, targetVals)  
    return vals / len(states2Check) 

def modify(r):
    rMax = np.max(r)
    rMin = np.min(r)
    mid = (rMax - rMin) / 2
    return r - mid
    


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

    if flags.FLAGS.checkDqnValues == "False":
        start_agent()
    else:
        check_dqn()


if __name__ == '__main__':
    print('Starting...')
    app.run(main)
