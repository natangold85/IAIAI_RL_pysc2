import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from algo_qtable import QTableParams
from algo_dqn import DQN_PARAMS
from algo_decisionMaker import DecisionMakerExperienceReplay
from utils_ttable import TransitionTable

from algo_dtn import DTN_PARAMS
from algo_dtn import DTN2
from algo_dtn import DTN
from algo_dtn import Filtered_DTN

from maze_game import SimpleMazeGame
from maze_game import MazeGame


def dtn_1LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        output = tf.contrib.layers.fully_connected(fc1, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_2LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        output = tf.contrib.layers.fully_connected(fc2, num_output)
        outputSoftmax = tf.nn.softmax(output, name="probability")

    return outputSoftmax

def dtn_2LayersFuncEmbedding(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):
        
        fe1State = tf.contrib.layers.fully_connected(inputLayerState, 256)
        fe1Action = tf.contrib.layers.fully_connected(inputLayerActions, 256)

        inputLayer = tf.concat([fe1State, fe1Action], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        output = tf.contrib.layers.fully_connected(fc2, num_output)
        outputSoftmax = tf.nn.softmax(output, name="probability")

    return outputSoftmax

def dtn_3LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)

        output = tf.contrib.layers.fully_connected(fc3, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax


def dtn_4LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)

        output = tf.contrib.layers.fully_connected(fc4, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_5LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        fc5 = tf.contrib.layers.fully_connected(fc4, 256)

        output = tf.contrib.layers.fully_connected(fc5, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax

def dtn_6LayersFunc(inputLayerState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerActions], 1)
        
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        fc5 = tf.contrib.layers.fully_connected(fc4, 256)
        fc6 = tf.contrib.layers.fully_connected(fc5, 256)

        output = tf.contrib.layers.fully_connected(fc6, num_output)
        outputSoftmax = tf.nn.softmax(output, name="softmax_tensor")

    return outputSoftmax


def dtn2_1LayersFunc(inputLayerState, inputLayerNextState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerNextState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)

        output = tf.contrib.layers.fully_connected(fc1, num_output)

    return output

def dtn2_2LayersFunc(inputLayerState, inputLayerNextState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerNextState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)

        output = tf.contrib.layers.fully_connected(fc2, num_output, activation_fn= tf.nn.sigmoid)

    return output


def dtn2_3LayersFunc(inputLayerState, inputLayerNextState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):

        inputLayer = tf.concat([inputLayerState, inputLayerNextState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        output = tf.contrib.layers.fully_connected(fc3, num_output)

    return output


def dtn2_4LayersFunc(inputLayerState, inputLayerNextState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):
        
        inputLayer = tf.concat([inputLayerState, inputLayerNextState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        output = tf.contrib.layers.fully_connected(fc4, num_output)

    return output

def dtn2_5LayersFunc(inputLayerState, inputLayerNextState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):
        
        inputLayer = tf.concat([inputLayerState, inputLayerNextState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        fc5 = tf.contrib.layers.fully_connected(fc4, 256)
        output = tf.contrib.layers.fully_connected(fc5, num_output)

    return output

def dtn2_6LayersFunc(inputLayerState, inputLayerNextState, inputLayerActions, num_output, scope):      
    with tf.variable_scope(scope):
        
        inputLayer = tf.concat([inputLayerState, inputLayerNextState, inputLayerActions], 1)
        fc1 = tf.contrib.layers.fully_connected(inputLayer, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 256)
        fc4 = tf.contrib.layers.fully_connected(fc3, 256)
        fc5 = tf.contrib.layers.fully_connected(fc4, 256)
        fc6 = tf.contrib.layers.fully_connected(fc5, 256)
        output = tf.contrib.layers.fully_connected(fc6, num_output)

    return output

class Simulator:
    def __init__(self, dirName = "maze_game", trials2Save = 100):
        self.illigalSolvedInModel = True
        #self.env = MazeGame()
        holes = []
        # holes.append([1,9])
        # holes.append([4,7])
        # holes.append([8,2])
        # holes.append([8,7])
        # holes.append([5,4])
        # holes.append([2,3])
        # holes.append([1,4])

        self.env = MazeGame(gridSize=10, holesCoord=holes)
        
        fullDir = "./" + dirName + "/"

        typeDecision = 'QLearningTable'
        params = QTableParams(self.env.stateSize, self.env.numActions)
        self.dqn = DecisionMakerExperienceReplay(typeDecision, params, decisionMakerName = "maze_game_dm_Time", resultFileName = "results_Time", directory = dirName, numTrials2Learn=trials2Save)            

        self.allDTN = []
        # dtn2Params = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn2_2LayersFunc)
        # self.allDTN.append(DTN2(dtn2Params, "dtn2_2Layers", directory = fullDir + 'maze_dtn2_2Layers/'))

        dtnParams = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_2LayersFunc)
        self.allDTN.append(DTN(dtnParams, "dtn_2Layers", directory = fullDir + 'maze_dtn_2Layers/'))

        # dtnParams3LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_3LayersFunc)
        # self.allDTN.append(DTN2(dtnParams3LTT, "dtn3Layers_Time", directory = fullDir + 'maze_dtn_3Layers_Time/'))

        # dtnParams4LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_4LayersFunc)
        # self.allDTN.append(DTN2(dtnParams4LTT, "dtn4Layers_Time", directory = fullDir + 'maze_dtn_4Layers_Time/'))

        # dtnParams5LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_5LayersFunc)
        # self.allDTN.append(DTN2(dtnParams5LTT, "dtn5Layers_Time", directory = fullDir + 'maze_dtn_5Layers_Time/'))

        # dtnParams6LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_6LayersFunc)
        # self.allDTN.append(DTN2(dtnParams6LTT, "dtn6Layers_Time", directory = fullDir + 'maze_dtn_6Layers_Time/'))

        self.transitionTable = TransitionTable(self.env.numActions, fullDir + "maze_ttable_cmp_Time")
    
    def ChooseAction(self, s):
        if self.illigalSolvedInModel:
            validActions = self.env.ValidActions(s)
            if np.random.uniform() > self.dqn.ExploreProb():
                valVec = self.dqn.ActionsValues(s)   
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
            toLearn = self.dqn.end_run(r,sumR,numSteps)

        self.TrainAccording2TTable()

        for dtn in self.allDTN:
            dtn.end_run(toLearn, False, numRuns)
        
        for i in range(len(self.allDTN)):
            self.trainDuration[i] += self.allDTN[i].LastTrainDuration()

    def TestTransition(self, numTests, num2Print = 1, fTest = None):
        
        mseResults = [0.0] * (len(self.allDTN) + 1)

        for i in range(numTests):
            s = self.env.randomState()
            validActions = self.env.ValidActions(s)
            a = np.random.choice(validActions)

            realDistribution = self.env.RealDistribution(s,a)
            for dtnIdx in range(len(self.allDTN)):
                if type(self.allDTN[dtnIdx]) is DTN:
                    outDTN = self.allDTN[dtnIdx].predict(s, a)[0]
                else:
                    outDTN = np.zeros(len(s), float)
                    for loc in range(len(s)):
                        s_ = np.zeros(len(s), int)
                        s_[loc] = 1
                        prob = self.allDTN[dtnIdx].TransitionProb(s, a, s_)
                        outDTN[loc] = prob

                    outDTN = outDTN / sum(outDTN)
                    


                mse = sum(pow(outDTN - realDistribution, 2)) 

                mseResults[dtnIdx] += mse / numTests
            
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

    def TrainAccording2TTable(self):
        states = list(self.transitionTable.table.keys())
        

        for sStr in states:
            if sStr != "TrialsData":
                s = np.fromstring(sStr.replace("[", "").replace("]", ""), dtype=int, sep=' ')
                actionCount = self.transitionTable.table[sStr][1]
                for a in range(self.env.numActions):
                    if actionCount[a] > 0:
                        label = self.CalcDistTTable(s,a)
                        
                        for dtn in self.allDTN:
                            if type(dtn) is DTN:
                                dtn.NewExperience(s, a, label)
                            else:
                                for loc in range(len(label)):
                                    s_ = np.zeros(len(label), int)
                                    s_[loc] = 1
                                    p = label[loc]

                                    dtn.NewExperience(s, a, s_, p)
                                
    
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
    
    dtnParams = DTN_PARAMS(10, 5, 0, 10, outputGraph=True, nn_Func=dtn_2LayersFuncEmbedding)
    dtn = DTN(dtnParams, "dtn_2Layers", directory = './test2/maze_dtn_2Layers/')

    print("\n\nfinished\n\n")
    exit()
    dirName = "maze_game"
    fMsePlotName = "./" + dirName + "/maze_dtn_layers.png"

    numRuns = 1000
    numRounds = 10
    numTrialsInRound = 20

    leg = ["s2p", "s2s_", "ttable"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

    
    sim = Simulator(dirName=dirName, trials2Save=numTrialsInRound)
    # trainDuration = []
    # saveDuration= []
    # for i in range(len(sim.allDTN)):
    #     trainDuration.append([])
    #     saveDuration.append([])

    mseAllResults = []
    t = np.arange(numRounds + 1) * numTrialsInRound
    minMseAllResults = []

    for i in range(len(leg)):
        minMseAllResults.append([100])

    print("\n\nfinished init!\n")
    for rnd in range(numRuns):
        mseResults = []
        sim.Reset()
        for i in range(numRounds):
            mse = sim.TestTransition(100)
            print("mse = ", mse)
            mseResults.append(mse)
            
            sim.Simulate(numTrialsInRound)
        
        mse = sim.TestTransition(100)
        mseResults.append(mse)
        mseAllResults.append(mseResults)

        mseResultsNp = np.array(mseResults)
        for i in range(len(mse)):
            currMinMse = minMseAllResults[i][-1]
            if (mse[i] < currMinMse):
                minMseAllResults[i] = mseResultsNp[:, i]

        print("\n\nfinished round #", rnd, end = '\n\n\n')

        if rnd > 0:
            mseAllResultsNp = np.array(mseAllResults)
            resultsMseAvg = np.average(mseAllResultsNp, axis=0)
            resultsMseStd = np.std(mseAllResultsNp, axis=0)

            fig = plt.figure(figsize=(19.0, 11.0))
            plt.subplot(2,2,1)
            for i in range(len(leg)):
                ub = resultsMseAvg[:,i] + resultsMseStd[:,i]
                lb = resultsMseAvg[:,i] - resultsMseStd[:,i]

                plot_mean_and_CI(resultsMseAvg[:,i], lb, ub, colors[i])

            plt.title("mse results for maze")
            plt.ylabel('mse')
            plt.legend(leg, loc='best')
            plt.xlabel('#trials')

            finalResults = mseAllResultsNp[:,-1,:]
            finalResultAvg = np.average(finalResults, axis=0)
            plt.subplot(2,2,2)
            idx = np.arange(len(finalResultAvg))

            plt.bar(idx, finalResultAvg, yerr = np.std(finalResults, axis=0))
            plt.xticks(idx, leg)
            plt.title("mse final results for maze")
            plt.ylabel('final mse')

            # best result:

            minMseAllResultsNp = np.matrix.transpose(np.array(minMseAllResults))
            plt.subplot(2,2,3)
            plt.plot(t, minMseAllResultsNp)
            plt.title("mse best results for maze")
            plt.ylabel('mse')
            plt.legend(leg, loc='best')
            plt.xlabel('#trials')

            finalResultAvgMin = minMseAllResultsNp[-1,:]
            plt.subplot(2,2,4)
            idx = np.arange(len(finalResultAvgMin))
            plt.bar(idx, finalResultAvgMin)
            plt.xticks(idx, leg)
            plt.title("mse final best results for maze")
            plt.ylabel('final mse')

            fig.savefig(fMsePlotName)

        # sim.SaveDTN()   
        # sumTrainDur, currSaveDur = sim.GetDTNDuration()
        
        # print(sumTrainDur)

        # avgTrainDur = np.array(sumTrainDur) / numRounds

        # print(avgTrainDur)

        # for i in range(len(avgTrainDur)):
        #     trainDuration[i].append(avgTrainDur[i])
        #     saveDuration[i].append(currSaveDur[i])

        # print("\n\nfinished round #", rnd, end = '\n\n\n')

        # allRuns = np.arange(rnd + 1)
        # trainDurationNP = np.matrix.transpose(np.array(trainDuration))
        # saveDurationNP = np.matrix.transpose(np.array(saveDuration))
        # if rnd > 0:
        #     fig = plt.figure(figsize=(19.0, 11.0))
        #     plt.subplot(2,1,1)
        #     plt.plot(allRuns, trainDurationNP)
        #     plt.title("train duration")
        #     plt.legend(leg, loc='best')
        #     plt.ylabel('[ms]')
        #     plt.xlabel('#runs')
        #     plt.subplot(2,1,2)
        #     plt.plot(allRuns, saveDurationNP)
        #     plt.title("save duration")
        #     plt.legend(leg, loc='best')
        #     plt.ylabel('[ms]')
        #     plt.xlabel('#runs')
        #     fig.savefig(fMsePlotName)
        


def Create():
    fullDir = "./maze_game/"
    params = DTN_PARAMS(1, 2 , 0, 1, nn_Func=dtn_2LayersFunc)
    dtn = DTN2(params, "dtn2Layers", directory = fullDir + 'maze_dtn_2Layers_Time/')
    
    return dtn

def Train(dtn, n = 100):
    valRange = np.array([-2, -1, 1, 2])

    for i in range(n):
        s = np.zeros(1, int)
        s[0] = np.random.choice(valRange)
        a = np.random.randint(0, 2)
        s_ = s.copy()
        if np.random.uniform() > 0.8:
            p = 0.8
            s_[0] = s_[0] + 1 if a == 0 else s_[0] - 1
        else:
            p = 0.2
        
        dtn.insert2Hist(s, a, s_, p)


    dtn.end_run()



def Test(dtn):
    testSet = []
    testSet.append(np.array([-2]))
    testSet.append(np.array([-1]))
    testSet.append(np.array([1]))
    testSet.append(np.array([2]))
    for i in range(4):
        s = testSet[i]
        a = 0
        s_ = s.copy()
        s_[0] += 1
        print(dtn.TransitionProb(s,a,s_))