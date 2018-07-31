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

class Maze:
    def __init__(self, gridSize):
        self.gridSize = gridSize

        self.action_doNothing = 0
        self.action_north = 1
        self.action_south = 2
        self.action_east = 3
        self.action_west = 4

        self.moves = {}
        self.moves[self.action_north] = [0,-1]
        self.moves[self.action_south] = [0,1]
        self.moves[self.action_east] = [1,0]
        self.moves[self.action_west] = [-1,0]

    def coord2Idx(self, coord):
        return coord[0] + coord[1] * self.gridSize
    
    def idx2Coord(self, idx):
        return [idx % self.gridSize, int(idx / self.gridSize)]

class SimpleMazeGame(Maze):
    def __init__(self):
        super(SimpleMazeGame, self).__init__(4)
        self.stateSize = self.gridSize * self.gridSize

        self.startingPntIdx = 0
        self.targetIdx = 3

        self.numActions = 5

        self.successInMove = 0.8
    

    def newGame(self):
        s = np.zeros(self.stateSize,dtype = int)
        s[self.startingPntIdx] = 1
        return s

    def step(self, s, a):
        loc = (s == 1).nonzero()[0][0]
        if loc == self.targetIdx:
            return s, 1.0, True

        if a == self.action_doNothing:
            return s.copy(), 0, False
        elif a < self.numActions:
            
            if np.random.uniform() < self.successInMove:
                coord = self.idx2Coord(loc)
                move = self.moves[a]
                for i in range(2):
                    toChange = coord[i] + move[i]
                    if toChange >= 0 and toChange < self.gridSize:
                        coord[i] = toChange

                s_ = np.zeros(self.stateSize,dtype = int)
                s_[self.coord2Idx(coord)] = 1
            else:
                s_ = s.copy()
        
        return s_, 0, False

    def ValidActions(self, s):
        return list(range(self.numActions))

    def randomState(self):
        s = np.zeros(self.stateSize,dtype = int)
        idx = np.random.randint(0, self.stateSize)
        s[idx] = 1
        return s

    def RealDistribution(self, s, a):
        dist = np.zeros(self.stateSize, float)
        loc = (s == 1).nonzero()[0][0]

        if loc == self.targetIdx:
            dist[loc] = 1.0
        elif a == self.action_doNothing:
            dist[loc] = 1.0
        else:
            coord = self.idx2Coord(loc)
            move = self.moves[a]
            for i in range(2):
                toChange = coord[i] + move[i]
                if toChange >= 0 and toChange < self.gridSize:
                    coord[i] = toChange

            newLoc = self.coord2Idx(coord)

            dist[newLoc] += self.successInMove
            dist[loc] += 1.0 - self.successInMove

        return dist

class MazeGame:
    def __init__(self):
        self.gridSize = 5
        self.stateSize = self.gridSize * self.gridSize + 1

        self.numSteps = 0
        self.maxSteps = 1000

        self.startingPntIdx = 0
        self.targetIdx = 24

        self.reward_InPenaltyLoc = -0.25
        self.reward_IlligalMove = -0.1
        self.numActions = 6
        self.action_doNothing = 0
        self.action_north = 1
        self.action_south = 2
        self.action_east = 3
        self.action_west = 4
        self.action_outFromPenalty = 5

        self.moves = {}
        self.moves[self.action_north] = [0,-1]
        self.moves[self.action_south] = [0,1]
        self.moves[self.action_east] = [1,0]
        self.moves[self.action_west] = [-1,0]

        self.holesIdx = []
        self.holesIdx.append(self.coord2Idx([3,2]))
        self.holesIdx.append(self.coord2Idx([1,3]))
        self.holesIdx.append(self.coord2Idx([2,2]))

        self.penaltyIdx = self.gridSize * self.gridSize

        self.successInMove = 0.8
        self.intoWormHole = 0.8

    def coord2Idx(self, coord):
        return coord[0] + coord[1] * self.gridSize
    
    def idx2Coord(self, idx):
        return [idx % self.gridSize, int(idx / self.gridSize)]

    def newGame(self):
        s = np.zeros(self.stateSize,dtype = int)
        s[self.startingPntIdx] = 1
        self.counterPenalty = 5
        self.numSteps = 0
        return s

    def InWormHole(self, loc):
        return loc in self.holesIdx

    def step(self, s, a):
        self.numSteps += 1
        loc = (s == 1).nonzero()[0][0]
        if loc == self.targetIdx:
            return s, 1.0, True
        if self.numSteps == self.maxSteps:
            return s, -1.0, True

        if loc == self.penaltyIdx:
            if self.counterPenalty == 0:
                return s.copy(), -1.0, True
            
            r = self.reward_InPenaltyLoc
            self.counterPenalty -= 1
        else:
            r = 0.0
        
        if a == self.action_doNothing:
            return s.copy(), r, False
        elif a < self.action_outFromPenalty:
            
            if loc != self.penaltyIdx and np.random.uniform() < self.successInMove:
                coord = self.idx2Coord(loc)
                move = self.moves[a]
                for i in range(2):
                    toChange = coord[i] + move[i]
                    if toChange >= 0 and toChange < self.gridSize:
                        coord[i] = toChange
                    else:
                        r = self.reward_IlligalMove

                s_ = np.zeros(self.stateSize,dtype = int)
                s_[self.coord2Idx(coord)] = 1
            else:
                s_ = s.copy()


        else:
            if loc == self.penaltyIdx and np.random.uniform() < self.successInMove:
                s_ = np.zeros(self.stateSize,dtype = int)
                s_[self.startingPntIdx] = 1
            else:
                s_ = s.copy()
        
        if self.InWormHole(loc):
            if np.random.uniform() < self.intoWormHole:
                s_ = np.zeros(self.stateSize,dtype = int)
                s_[self.penaltyIdx] = 1
        
        return s_, r, False

    def ValidActions(self, s):
        loc = (s == 1).nonzero()[0][0]     
        if loc == self.penaltyIdx:
            return list(range(self.numActions))
        else:
            valid = [self.action_doNothing, self.action_outFromPenalty]
            coord = self.idx2Coord(loc)
            for a in range(self.action_north, self.action_outFromPenalty):
                validAction = True
                for i in range(2):
                    l = coord[i] + self.moves[a][i]
                    if l < 0 or l >= self.gridSize:
                        validAction = False
                if validAction:
                    valid.append(a)
            return valid

    def RobCoord(self, state):
        loc = (state == 1).nonzero()[0][0]
        return self.idx2Coord(loc)

    def randomState(self):
        s = np.zeros(self.stateSize,dtype = int)
        idx = np.random.randint(0, self.stateSize)
        s[idx] = 1
        return s

    def RealDistribution(self, s, a):
        dist = np.zeros(self.stateSize, float)
        loc = (s == 1).nonzero()[0][0]

        if loc == self.targetIdx:
            dist[loc] = 1.0
        elif a == self.action_doNothing:
            dist[loc] = 1.0
        elif a < self.action_outFromPenalty:
            if loc != self.penaltyIdx:
                coord = self.idx2Coord(loc)
                move = self.moves[a]
                for i in range(2):
                    toChange = coord[i] + move[i]
                    if toChange >= 0 and toChange < self.gridSize:
                        coord[i] = toChange

                newLoc = self.coord2Idx(coord)

                dist[newLoc] += self.successInMove
                dist[loc] += 1.0 - self.successInMove
            else:
                dist[loc] = 1.0
        else:
            if loc == self.penaltyIdx:
                dist[self.penaltyIdx] = self.successInMove
                dist[loc] = 1.0 - self.successInMove
            else:
                dist[loc] = 1.0

        # check if possible outcome in worm hole:
        idxList = np.argwhere(dist > 0)
        for idx in idxList:
            newLoc = idx[0]
            if self.InWormHole(newLoc):
                whole = dist[newLoc]
                dist[newLoc] *= 1 - self.intoWormHole
                dist[self.penaltyIdx] += whole * self.intoWormHole

        return dist

    def PrintState(self, s):
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                print(s[x + y * self.gridSize], end = ', ')
            print("|")
        print(s[self.penaltyIdx])

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
        typeD = 'QReplay'
        
        fullDir = "./" + dirName + "/"

        if typeD == "NN":
            params = DQN_PARAMS(self.env.stateSize, self.env.numActions)
            self.dqn = LearnWithReplayMngr(typeD, params, dqnName = "dqn", qTableName= "qTable", resultFileName = "results", directory = dirName, numTrials2Learn=trials2Save)
        else:
            params = QTableParams(self.env.stateSize, self.env.numActions)
            self.dqn = LearnWithReplayMngr(typeD, params, dqnName = "dqn", qTableName= "qTable", resultFileName = "results", directory = dirName, numTrials2Learn=trials2Save)            

        self.allDTN = []
        dtnParams1LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_1LayersFunc)
        self.allDTN.append(DTN(dtnParams1LTT, "dtn1Layers_TT", directory = fullDir + 'maze_dtn_1Layers_TT/'))

        dtnParams2LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True)
        self.allDTN.append(DTN(dtnParams2LTT, "dtn2Layers_TT", directory = fullDir + 'maze_dtn_2Layers_TT/'))

        dtnParams3LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_3LayersFunc)
        self.allDTN.append(DTN(dtnParams3LTT, "dtn3Layers_TT", directory = fullDir + 'maze_dtn_3Layers_TT/'))

        dtnParams4LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_4LayersFunc)
        self.allDTN.append(DTN(dtnParams4LTT, "dtn4Layers_TT", directory = fullDir + 'maze_dtn_4Layers_TT/'))

        dtnParams5LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_5LayersFunc)
        self.allDTN.append(DTN(dtnParams5LTT, "dtn5Layers_TT", directory = fullDir + 'maze_dtn_5Layers_TT/'))

        dtnParams6LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_6LayersFunc)
        self.allDTN.append(DTN(dtnParams6LTT, "dtn6Layers_TT", directory = fullDir + 'maze_dtn_6Layers_TT/'))

        dtnParams7LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_7LayersFunc)
        self.allDTN.append(DTN(dtnParams7LTT, "dtn7Layers_TT", directory = fullDir + 'maze_dtn_7Layers_TT/'))

        self.transitionTable = TransitionTable(self.env.numActions, fullDir + "maze_ttable_cmp")
    
    def ChooseAction(self, s):
        if self.illigalSolvedInModel:
            validActions = self.env.ValidActions(s)
            if np.random.uniform() > self.dqn.ExploreProb():
                valVec = self.dqn.actionValuesVec(s)   
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


def plot_mean_and_CI(mean, lb, ub, color_mean=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb, color=color_mean, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

if __name__ == "__main__":
    
    dirName = "maze_game"
    fMsePlotName = "./" + dirName + "/maze_mse_results.png"

    numRuns = 1000
    numRounds = 20
    numTrialsInRound = 20

    leg = ["1Layers", "2Layers", "3Layers", "4Layers", "5Layers", "6Layers", "7Layers", "ttable"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

    sim = Simulator(dirName=dirName, trials2Save=numTrialsInRound)
    
    minMseAllResults = []
    for i in range(len(leg)):
        minMseAllResults.append([100])
    
    mseAllResults = []

    t = np.arange(numRounds + 1) * numTrialsInRound

    for rnd in range(numRuns):
        
        mseResults = []

        sim.Reset()
        for i in range(numRounds):
            mse = sim.TestTransition(500)
            mseResults.append(mse)


            sim.Simulate(numTrialsInRound)

        mse = sim.TestTransition(500)
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




