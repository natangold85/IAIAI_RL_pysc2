import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from algo_dqn import DQN_PARAMS
from algo_dqn import DQN

import random
import math

from absl import app
from absl import flags
import sys

# possible agents :
from agent_super import SuperAgent

flags.DEFINE_string("train", "none", "Which agent to train.")
flags.DEFINE_string("directory", "none", "run directory")
flags.DEFINE_string("useGpu", "True", "Which device to run nn on.")
flags.DEFINE_string("episodes2Train", "all", "num of episodes to train.")
flags.DEFINE_string("trainStart", "0", "num of episodes to train.")
flags.DEFINE_string("saveLearning", "False", "if to save offline learning.")
flags.DEFINE_string("newLearning", "False", "reset agent learning")


def StateValues(decisionMaker, numStates = 100):
        
    allValues = []
    for i in range(numStates):
        s = decisionMaker.DrawStateFromHist()
        values = decisionMaker.ActionsValues(s)
        allValues.append(values)

    avgValues = np.average(allValues,axis = 0)
    return avgValues

def FullTrain(decisionMaker, newLearning, saveLearning):  
    episodes2Train = flags.FLAGS.episodes2Train
    if episodes2Train == "all":
        episodes2Train = math.inf
    else:
        episodes2Train = int(episodes2Train)

    print("get history")
    allHist = decisionMaker.historyMngr.GetAllHist().copy()
    
    print("reset data")
    if newLearning:
        decisionMaker.ResetAllData(resetHistory=False)  
    decisionMaker.ResetHistory(dump2Old=False)
    # init history mngr and resultFile without saving to files
    print("start learning...\n\n")

    decisionMaker.historyMngr.__init__(decisionMaker.params)
    decisionMaker.resultFile = None
    history = decisionMaker.AddHistory()

    startTrainEpisodeNums = list(map(int, flags.FLAGS.trainStart.split(',')))

    episodeNum = 0
    
    idxHist = 0
    rTerminal = []
    rTerminalSum = []

    allValues = []
    allValuesEpNum = []

    for startEpisodeNum in startTrainEpisodeNums:
        episodeInTrain = 0
        cotinueTrain = True
        while cotinueTrain:
            terminal = False
            steps = 0
            while not terminal:
                s = allHist["s"][idxHist]
                a = allHist["a"][idxHist]
                r = allHist["r"][idxHist]
                s_ = allHist["s_"][idxHist]
                terminal = allHist["terminal"][idxHist]
                
                history.learn(s, a, r, s_, terminal)
                steps += 1
                idxHist += 1
            
            rTerminal.append(r)
            decisionMaker.end_run(r, 0, steps)
            episodeNum += 1
            if episodeNum >= startEpisodeNum:
                episodeInTrain += 1
                if episodeInTrain >= episodes2Train:
                    cotinueTrain = False                    

            if decisionMaker.trainFlag:
                if episodeNum >= startEpisodeNum:
                    print("\n\t\t episode num =", episodeNum, "episode in  train =", episodeInTrain)
                    decisionMaker.Train()
                    print("training agent episode #", episodeNum)
                    stateValues = StateValues(decisionMaker)
                    allValues.append(stateValues)
                else:
                    vals = np.zeros(decisionMaker.params.numActions)
                    vals.fill(np.nan)
                    allValues.append(vals)

                decisionMaker.trainFlag = False
                allValuesEpNum.append(episodeNum)


            if len(rTerminal) > 200:
                rTerminal.pop(0)

            if len(rTerminal) == 200:
                rTerminalSum.append(np.average(rTerminal))

    if saveLearning:
        decisionMaker.decisionMaker.Save()    

    plt.subplot(2,2,1)
    plt.plot(rTerminalSum)

    plt.subplot(2,2,2)
    plt.plot(allValuesEpNum, allValues)

    plt.subplot(2,2,3)
    plt.plot(allValuesEpNum, np.average(allValues, axis = 1))
    plt.show()

if __name__ == "__main__":
    flags.FLAGS(sys.argv)

    directoryName = flags.FLAGS.directory
    
    configDict = eval(open(directoryName + "\config.txt", "r+").read())
    configDict["directory"] = directoryName
    agent = SuperAgent(configDict=configDict)

    useGpu = bool(eval(flags.FLAGS.useGpu))
    saveLearning = bool(eval(flags.FLAGS.saveLearning))
    newLearning = bool(eval(flags.FLAGS.newLearning))

    if not useGpu:
        # run from cpu
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    

    agentName = flags.FLAGS.train

    decisionMaker = agent.decisionMaker.GetDecisionMakerByName(agentName)
    
    if decisionMaker != None:  
        print("start training...\n\n")
        FullTrain(decisionMaker, newLearning, saveLearning)
    else:
        print("error in loading the nn")


        
    
