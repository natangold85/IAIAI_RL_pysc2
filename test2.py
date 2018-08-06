import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from utils_qtable import QTableParams
from utils_dqn import DQN_PARAMS
from utils_decisionMaker import LearnWithReplayMngr
from utils_ttable import TransitionTable

from utils_dtn import DTN_PARAMS
from utils_dtn import DTN
from utils_dtn import Filtered_DTN

from maze_game import SimpleMazeGame

def dtn_3LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        el1 = tf.contrib.layers.fully_connected(inputLayerState, 256)
        middleLayer = tf.concat([el1, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)

        output = tf.contrib.layers.fully_connected(fc2, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_1LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)

        output = tf.contrib.layers.fully_connected(fc1, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_4LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        el1 = tf.contrib.layers.fully_connected(inputLayerState, 256)
        middleLayer = tf.concat([el1, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        output = tf.contrib.layers.fully_connected(fc3, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_5LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        el1 = tf.contrib.layers.fully_connected(inputLayerState, 256)
        middleLayer = tf.concat([el1, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        output = tf.contrib.layers.fully_connected(fc4, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_6LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        el1 = tf.contrib.layers.fully_connected(inputLayerState, 256)
        middleLayer = tf.concat([el1, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        fc5 = tf.contrib.layers.fully_connected(fc4, 256)
        output = tf.contrib.layers.fully_connected(fc5, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_7LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        el1 = tf.contrib.layers.fully_connected(inputLayerState, 256)
        middleLayer = tf.concat([el1, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(middleLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        fc5 = tf.contrib.layers.fully_connected(fc4, 256)
        fc6 = tf.contrib.layers.fully_connected(fc5, 256)
        output = tf.contrib.layers.fully_connected(fc6, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

class Simulator:
    def __init__(self, dirName = "maze_game", trials2Save = 100):
        self.illigalSolvedInModel = True
        #self.env = MazeGame()
        self.env = SimpleMazeGame()
        
        fullDir = "./" + dirName + "/"

        typeDecision = 'QLearningTable'
        params = QTableParams(self.env.stateSize, self.env.numActions)
        self.dqn = LearnWithReplayMngr(typeDecision, params, decisionMakerName = "maze_game_dm_Time", resultFileName = "results_Time", directory = dirName, numTrials2Learn=trials2Save)            

        self.allDTN = []
        dtnParams1LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_1LayersFunc)
        self.allDTN.append(DTN(dtnParams1LTT, "dtn1Layers_Time", directory = fullDir + 'maze_dtn_1Layers_Time/'))

        dtnParams2LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True)
        self.allDTN.append(DTN(dtnParams2LTT, "dtn2Layers_Time", directory = fullDir + 'maze_dtn_2Layers_Time/'))

        dtnParams3LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_3LayersFunc)
        self.allDTN.append(DTN(dtnParams3LTT, "dtn3Layers_Time", directory = fullDir + 'maze_dtn_3Layers_Time/'))

        dtnParams4LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_4LayersFunc)
        self.allDTN.append(DTN(dtnParams4LTT, "dtn4Layers_Time", directory = fullDir + 'maze_dtn_4Layers_Time/'))

        dtnParams5LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_5LayersFunc)
        self.allDTN.append(DTN(dtnParams5LTT, "dtn5Layers_Time", directory = fullDir + 'maze_dtn_5Layers_Time/'))

        dtnParams6LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_6LayersFunc)
        self.allDTN.append(DTN(dtnParams6LTT, "dtn6Layers_Time", directory = fullDir + 'maze_dtn_6Layers_Time/'))

        dtnParams7LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_7LayersFunc)
        self.allDTN.append(DTN(dtnParams7LTT, "dtn7Layers_Time", directory = fullDir + 'maze_dtn_7Layers_Time/'))

        self.transitionTable = TransitionTable(self.env.numActions, fullDir + "maze_ttable_cmp_Time")
    
    def ChooseAction(self, s):
        if self.illigalSolvedInModel:
            validActions = self.env.ValidActions(s)
            if np.random.uniform() > self.dqn.ExploreProb():
                valVec = self.dqn.ActionValuesVec(s)   
                random.shuffle(validActions)
                validVal = valVec[validActions]
                action = validActions[validVal.argmax()]
            else:
                action = np.random.choice(validActions) 
        else:
            action = self.dqn.choose_action(s)
        return action

    def Simulate(self, numRuns = 1):
        for i in range(numRuns):
            s = self.env.newGame()
            
            t = False
            sumR = 0
            numSteps = 0
            while not t:
                a = self.ChooseAction(s)
                s_, r, t = self.env.step(s,a)
                
                self.dqn.learn(s, a, r, s_, t)
                self.transitionTable.learn(str(s), a, str(s_))
                sumR += r
                numSteps += 1
                s = s_
            
            self.transitionTable.end_run(True)
            self.dqn.end_run(r,sumR,numSteps)

        for dtn in self.allDTN:
            self.TrainAccording2TTable(dtn, numRuns)
            dtn.end_run(False, numRuns)
        
        for i in range(len(self.allDTN)):
            self.trainDuration[i] += self.allDTN[i].LastTrainDuration()

    def TestTransition(self, numTests, num2Print = 1, fTest = None):
        
        mseResults = [0.0] * (len(self.allDTN) + 1)

        for i in range(numTests):
            s = self.env.randomState()
            validActions = self.env.ValidActions(s)
            a = np.random.choice(validActions)

            realDistribution = self.env.RealDistribution(s,a)
            for i in range(len(self.allDTN)):

                outDtn = self.allDTN[i].predict(s,a)
                mse = sum(pow(outDtn[0] - realDistribution, 2)) 
                mseResults[i] += mse / numTests
            
            outTransitionTable = self.CalcDistTTable(s,a)                        
            mseTable = sum(pow(outTransitionTable - realDistribution, 2))
            mseResults[len(self.allDTN)] += mseTable / numTests

        
        return mseResults

    def CalcDistTTable(self,s,a):
        outTransitionTable = np.zeros(self.env.stateSize, dtype = float)

        validTTable = False
        sStr = str(s)
        if sStr in self.transitionTable.table:
            transition = self.transitionTable.table[sStr][0]
            actionCount = self.transitionTable.table[sStr][1]
            if actionCount[a] > 0:
                validTTable = True

                sumTransitionCount = actionCount[a]
                states = list(transition.index)
                for s_ in states:
                    currStateCount = transition.ix[s_,a]
                    if currStateCount > 0:
                        modS_ = s_.replace("[", "")
                        modS_ = modS_.replace("]", "")
                        s_Array = np.fromstring(modS_, dtype=int, sep=' ')
                        loc = (s_Array == 1).nonzero()[0][0] 
                        outTransitionTable[loc] = currStateCount / sumTransitionCount
        
        if not validTTable:
            outTransitionTable += 1.0 / self.env.stateSize

        return outTransitionTable

    def TrainAccording2TTable(self, dtn, numTrains = 1):
        states = list(self.transitionTable.table.keys())
        
        sLearn = []
        aLearn = []
        s_Learn = []
        for i in range(numTrains):
            for sStr in states:
                if sStr != "TrialsData":
                    s = np.fromstring(sStr.replace("[", "").replace("]", ""), dtype=int, sep=' ')
                    transition = self.transitionTable.table[sStr][0]
                    actionCount = self.transitionTable.table[sStr][1]
                    for a in range(self.env.numActions):
                        if actionCount[a] > 0:
                            label = self.CalcDistTTable(s,a)
                            sLearn.append(s)
                            aLearn.append(a)
                            s_Learn.append(label)
        
        if len(aLearn) >= dtn.params.batchSize:
            dtn.learn(np.array(sLearn), np.array(aLearn), np.array(s_Learn))
    
    def Reset(self):
        self.dqn.ResetAllData()
        for dtn in self.allDTN:
            dtn.Reset()

        self.transitionTable.Reset()

        self.trainDuration = []
        self.saveDuration = []
        for i in range(len(self.allDTN)):
            self.trainDuration.append(0.0)
            self.saveDuration.append(0.0)

    def GetDTNDuration(self):
        return self.trainDuration, self.saveDuration

    def SaveDTN(self):
        for i in range(len(self.allDTN)):
            self.allDTN[i].save_network()
            self.saveDuration[i] += self.allDTN[i].LastSaveDuration()


def plot_mean_and_CI(mean, lb, ub, color_mean=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb, color=color_mean, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

if __name__ == "__main__":
    
    dirName = "maze_game"
    fMsePlotName = "./" + dirName + "/maze_time_duration.png"

    numRuns = 1000
    numRounds = 20
    numTrialsInRound = 20

    leg = ["1Layers", "2Layers", "3Layers", "4Layers", "5Layers", "6Layers", "7Layers"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

    
    sim = Simulator(dirName=dirName, trials2Save=numTrialsInRound)
    trainDuration = []
    saveDuration= []
    for i in range(len(sim.allDTN)):
        trainDuration.append([])
        saveDuration.append([])

    for rnd in range(numRuns):
          
        sim.Reset()
        for i in range(numRounds):
            sim.Simulate(numTrialsInRound)
        
        sim.SaveDTN()   
        sumTrainDur, currSaveDur = sim.GetDTNDuration()
        
        print(sumTrainDur)

        avgTrainDur = np.array(sumTrainDur) / numRounds

        print(avgTrainDur)

        for i in range(len(avgTrainDur)):
            trainDuration[i].append(avgTrainDur[i])
            saveDuration[i].append(currSaveDur[i])

        print("\n\nfinished round #", rnd, end = '\n\n\n')

        allRuns = np.arange(rnd + 1)
        trainDurationNP = np.matrix.transpose(np.array(trainDuration))
        saveDurationNP = np.matrix.transpose(np.array(saveDuration))
        if rnd > 0:
            fig = plt.figure(figsize=(19.0, 11.0))
            plt.subplot(2,1,1)
            plt.plot(allRuns, trainDurationNP)
            plt.title("train duration")
            plt.legend(leg, loc='best')
            plt.ylabel('[ms]')
            plt.xlabel('#runs')
            plt.subplot(2,1,2)
            plt.plot(allRuns, saveDurationNP)
            plt.title("save duration")
            plt.legend(leg, loc='best')
            plt.ylabel('[ms]')
            plt.xlabel('#runs')
            fig.savefig(fMsePlotName)

        #     fig = plt.figure(figsize=(19.0, 11.0))
        #     plt.subplot(2,2,1)
        #     for i in range(len(leg)):
        #         ub = resultsMseAvg[:,i] + resultsMseStd[:,i]
        #         lb = resultsMseAvg[:,i] - resultsMseStd[:,i]

        #         plot_mean_and_CI(resultsMseAvg[:,i], lb, ub, colors[i])

        #     plt.title("mse results for maze")
        #     plt.ylabel('mse')
        #     plt.legend(leg, loc='best')
        #     plt.xlabel('#trials')

        #     finalResults = mseAllResultsNp[:,-1,:]
        #     finalResultAvg = np.average(finalResults, axis=0)
        #     plt.subplot(2,2,2)
        #     idx = np.arange(len(finalResultAvg))

        #     plt.bar(idx, finalResultAvg, yerr = np.std(finalResults, axis=0))
        #     plt.xticks(idx, leg)
        #     plt.title("mse final results for maze")
        #     plt.ylabel('final mse')

        #     # best result:

        #     minMseAllResultsNp = np.matrix.transpose(np.array(minMseAllResults))
        #     plt.subplot(2,2,3)
        #     plt.plot(t, minMseAllResultsNp)
        #     plt.title("mse best results for maze")
        #     plt.ylabel('mse')
        #     plt.legend(leg, loc='best')
        #     plt.xlabel('#trials')

        #     finalResultAvgMin = minMseAllResultsNp[-1,:]
        #     plt.subplot(2,2,4)
        #     idx = np.arange(len(finalResultAvgMin))
        #     plt.bar(idx, finalResultAvgMin)
        #     plt.xticks(idx, leg)
        #     plt.title("mse final best results for maze")
        #     plt.ylabel('final mse')

        #     fig.savefig(fMsePlotName)




