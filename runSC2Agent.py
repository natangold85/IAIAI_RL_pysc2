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

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

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
flags.DEFINE_string("plotDiff", "False", "Which agent to check.")

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
        t = threading.Thread(target=run_thread, args=thread_args, daemon=True)
        t.setName("GameThread_" + str(idx))

        threads.append(t)
        t.start()
        time.sleep(5)
        show = False
        idx += 1


    runThreadAlive = True
    threading.current_thread().setName("TrainThread")
    
    dir2Save = "./" + dm_Types["directory"] + "/nnGraphs/"
    if not os.path.isdir(dir2Save):
        os.makedirs(dir2Save)

    while runThreadAlive:
        # train  when flag of training is on
        numTrials2Learn = decisionMaker.TrainAll()
        if numTrials2Learn >= 0:
            create_nnGraphs(agents[0], trainList[0], numTrials2Learn, saveGraphs=True, dir2Save = dir2Save)

        time.sleep(1)
        
        # if at least one thread is alive continue running
        runThreadAlive = False

        for t in threads:
            isAlive = t.isAlive()
            if isAlive:
                runThreadAlive = True
            else:
                t.join() 

        # global NUM_CRASHES
        # if NUM_CRASHES > NUM_CRASHES_2_RESTART or not RUN:
        #     print("\n\ninit all sc2 games\n\n")
        #     os.system('taskkill /f /im SC2_x64.exe')
        #     NUM_CRASHES = 0

    
def create_nnGraphs(superAgent, agent2Check, numTrials = 0, saveGraphs = False, showGraphs = False, dir2Save = "./"):
    figCurr = None
    figTarget = None

    figCurrDiff = None
    figTargetDiff = None
    
    allStatesIdx = flags.FLAGS.stateIdx2Check.split(",")
        
    idxX = int(allStatesIdx[0])
    idxY = int(allStatesIdx[1])
    

    agent = superAgent.GetAgentByName(agent2Check)
    dm = agent.GetDecisionMaker()

    xName = agent.StateIdx2Str(idxX)
    yName = agent.StateIdx2Str(idxY)

    actions2Check = flags.FLAGS.actions2Check.split(",")    
    
    actionsPoints = {}
    actionsPointsTarget = {}

    actionsPoints = {}
    actionsPointsTarget = {}

    # extracting nn vals for current nn and target nn

    for a in range(len(actions2Check)):
        actions2Check[a] = int(actions2Check[a])
        actionsPoints[actions2Check[a]] = [[], [], [], []]
        actionsPointsTarget[actions2Check[a]] = [[], [], [], []]

    sizeHist = len(dm.historyMngr.transitions["a"])
    for i in range(min(sizeHist, 20000)):
        s = dm.DrawStateFromHist()
        # s[idxY] = round((s[idxY] * 10) / 624)

        vals = dm.ActionValuesVec(s)
        valsTarget = dm.ActionValuesVec(s, "target")

        for a in actions2Check:
            add_point(s[idxX], s[idxY], vals[a], actionsPoints[a])
            add_point(s[idxX], s[idxY], valsTarget[a], actionsPointsTarget[a])

    # calculating avg val
    maxCurr = -1.0
    minCurr = 1.0

    maxTarget = -1.0
    minTarget = 1.0
    for a in actions2Check:
        for i in range(len(actionsPoints[a][0])):
            actionsPoints[a][3][i] = np.average(np.array(actionsPoints[a][2][i])) 
            maxCurr = max(maxCurr, actionsPoints[a][3][i])
            minCurr = min(minCurr, actionsPoints[a][3][i])

        for i in range(len(actionsPointsTarget[a][0])):
            actionsPointsTarget[a][3][i] = np.average(np.array(actionsPointsTarget[a][2][i])) 
            maxTarget = max(maxTarget, actionsPointsTarget[a][3][i])
            minTarget = min(minTarget, actionsPointsTarget[a][3][i])

    
    numRows = math.ceil(len(actions2Check) / 2)
    idxPlot = 1

    figCurr = plt.figure(figsize=(19.0, 11.0))
    plt.suptitle("action evaluation - current: (#trials = " + str(numTrials) + ")")
    for a in actions2Check:
        x = np.array(actionsPoints[a][0])
        y = np.array(actionsPoints[a][1])
        z = np.array(actionsPoints[a][3])
        ax = figCurr.add_subplot(numRows, 2, idxPlot)
        img = plotImg(ax, x, y, z, xName, yName, "values for action = " + agent.Action2Str(a), minZ=minCurr, maxZ=maxCurr)
        figCurr.colorbar(img, shrink=0.4, aspect=5)
        idxPlot += 1
    
    idxPlot = 1

    figTarget = plt.figure(figsize=(19.0, 11.0))
    plt.suptitle("action evaluation - target: (#trials = " + str(numTrials) + ")")
    for a in actions2Check:
        x = np.array(actionsPointsTarget[a][0])
        y = np.array(actionsPointsTarget[a][1])
        z = np.array(actionsPointsTarget[a][3])
        ax = figTarget.add_subplot(numRows, 2, idxPlot)
        img = plotImg(ax, x, y, z, xName, yName, "values for action = " + agent.Action2Str(a), minZ=minTarget, maxZ=maxTarget)
        figTarget.colorbar(img, shrink=0.4, aspect=5)
        idxPlot += 1

    plotDiff = eval(flags.FLAGS.plotDiff)
    if plotDiff:
        numRows = math.ceil(len(actions2Check) * (len(actions2Check) - 1) / 2)

        figCurrDiff = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle("differrence in action values - current: (#trials = " + str(numTrials) + ")")
        idxPlot = 1

        for a1Idx in range(len(actions2Check)):
            a1 = actions2Check[a1Idx]
            x = np.array(actionsPoints[a1][0])
            y = np.array(actionsPoints[a1][1])

            for a2Idx in range(a1Idx + 1, len(actions2Check)):
                a2 = actions2Check[a2Idx]
                z1 = np.array(actionsPoints[a1][3])
                z2 = np.array(actionsPoints[a2][3])
                zDiff = z2 - z1
                maxZ = np.max(np.abs(zDiff))
                ax = figCurrDiff.add_subplot(numRows, 2, idxPlot)
                title = "values for differrence = " + agent.Action2Str(a1) + " - " + agent.Action2Str(a2)
                img = plotImg(ax, x, y, zDiff, xName, yName, title, minZ=-maxZ, maxZ=maxZ)
                figCurrDiff.colorbar(img, shrink=0.4, aspect=5)
                idxPlot += 1



        figTargetDiff = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle("differrence in action values - target: (#trials = " + str(numTrials) + ")")
        idxPlot = 1

        for a1Idx in range(len(actions2Check)):
            a1 = actions2Check[a1Idx]
            x = np.array(actionsPointsTarget[a1][0])
            y = np.array(actionsPointsTarget[a1][1])

            for a2Idx in range(a1Idx + 1, len(actions2Check)):
                a2 = actions2Check[a2Idx]
                z1 = np.array(actionsPointsTarget[a1][3])
                z2 = np.array(actionsPointsTarget[a2][3])
                zDiff = z2 - z1
                maxZ = np.max(np.abs(zDiff))
                ax = figTargetDiff.add_subplot(numRows, 2, idxPlot, projection='3d')
                title = "values for differrence = " + agent.Action2Str(a1) + " - " + agent.Action2Str(a2)
                img = plotImg(ax, x, y, zDiff, xName, yName, title, minZ=-maxZ, maxZ=maxZ)
                figTargetDiff.colorbar(img, shrink=0.4, aspect=5)
                idxPlot += 1
    
    if saveGraphs:
        if figCurr != None:
            figCurr.savefig(dir2Save + "currentDQN_" + str(numTrials))
        if figTarget != None:
            figTarget.savefig(dir2Save + "targetDQN_" + str(numTrials))

        if figCurrDiff != None:
            figCurrDiff.savefig(dir2Save + "currentDQNDiff_" + str(numTrials))
        if figTargetDiff != None:
            figTargetDiff.savefig(dir2Save + "targetDQNDiff_" + str(numTrials))

    if showGraphs:
        plt.show()

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
    create_nnGraphs(superAgent, agent2Check, showGraphs=True)


def avg_values(dm, states, targetVals=False):
    vals = []
    for s in states:
        vals.append(dm.ActionValuesVec(s, targetVals))

    return np.average(vals, axis=0)

def add_point(x, y, val, actionVec):
    for i in range(len(actionVec[0])):
        if x == actionVec[0][i] and y == actionVec[1][i]:
            actionVec[2][i].append(val)
            return
    
    actionVec[0].append(x)
    actionVec[1].append(y)
    actionVec[2].append([val])
    actionVec[3].append(0)

def plot3d(ax, x, y, z, xName, yName, zName, title, minZ = None, maxZ = None):
    if minZ == None:
        minZ = np.min(z)
    if maxZ == None:
        maxZ = np.max(z)

    min_radius = 0.25
    
    tri = mtri.Triangulation(x, y)

    xmid = x[tri.triangles].mean(axis=1)
    ymid = y[tri.triangles].mean(axis=1)
    mask = np.where(xmid**2 + ymid**2 < min_radius**2, 1, 0)
    tri.set_mask(mask)

    surf = ax.plot_trisurf(tri, z, cmap=plt.cm.coolwarm, norm=colors.Normalize(vmin=minZ, vmax=maxZ))
    if np.max(x) - np.min(x) < 10:
        xTick = np.arange(np.min(x), np.max(x))
    else:
        xTick = np.arange(np.min(x), np.max(x), 4)

    if np.max(y) - np.min(y) < 10:
        yTick = np.arange(np.min(y), np.max(y))
    else:
        yTick = np.arange(np.min(y), np.max(y), 4)


    ax.set_xticks(xTick)
    ax.set_yticks(yTick)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_zlabel(zName)
    ax.set_title(title)

    return surf

def plotImg(ax, x, y, z, xName, yName, title, minZ = None, maxZ = None):
    imSizeX = np.max(x) - np.min(x) + 1
    imSizeY = np.max(y) - np.min(y) + 1

    offsetX = np.min(x)
    offsetY = np.min(y)

    mat = np.zeros((imSizeY, imSizeX), dtype=float)
    mat.fill(np.nan)

    for i in range(len(x)):
        #print("[", x[i], y[i], "] =", z[i])
        mat[y[i] + offsetY, x[i] + offsetX] = z[i]

    if minZ == None:
        minZ = np.min(mat)
    if maxZ == None:
        maxZ = np.max(mat)

    img = ax.imshow(mat, cmap=plt.cm.coolwarm, vmin=minZ, vmax=maxZ)

    if np.max(x) - np.min(x) < 10:
        xTick = np.arange(np.min(x), np.max(x))
    else:
        xTick = np.arange(np.min(x), np.max(x), 4)

    if np.max(y) - np.min(y) < 10:
        yTick = np.arange(np.min(y), np.max(y))
    else:
        yTick = np.arange(np.min(y), np.max(y), 4)

    ax.set_xticks(xTick)
    ax.set_yticks(yTick)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_title(title)

    return img


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
