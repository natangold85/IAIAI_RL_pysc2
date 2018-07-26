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
class Simulator:
    def __init__(self, dirName = "maze_game",trials2Save = 100):
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
        
        #dtnParams = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True)
        #self.dtnWith2LayersFunc = DTN(dtnParams, "dtn2Layers", directory = fullDir + 'maze_dtn_2Layers/')
        
        dtnParams3L = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_3LayersFunc)
        self.dtn3Layers = DTN(dtnParams3L, "dtn3Layers", directory = fullDir + 'maze_dtn_3Layers/')
        
        dtnParams3LTT = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_3LayersFunc)
        self.dtn3LayersWithTTable = DTN(dtnParams3LTT, "dtn3Layers_TT", directory = fullDir + 'maze_dtn_3Layers_TT/')

        dtnParams4L = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_4LayersFunc)
        self.dtn4Layers = DTN(dtnParams4L, "dtn4Layers", directory = fullDir + 'maze_dtn_4Layers/')

        dtnParams5L = DTN_PARAMS(self.env.stateSize, self.env.numActions, 0, self.env.stateSize, outputGraph=True, nn_Func=dtn_5LayersFunc)
        self.dtn5Layers = DTN(dtnParams5L, "dtn5Layers", directory = fullDir + 'maze_dtn_5Layers/')

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
            
            s,a,_,s_,_ = self.dqn.ExperienceReplay()
            self.dtn3Layers.learn(s, a, s_)
            self.dtn4Layers.learn(s, a, s_)
            self.dtn5Layers.learn(s, a, s_)

            for i in range(len(a)): 
                self.transitionTable.learn(str(s[i]), a[i], str(s_[i]))

            self.TrainAccording2TTable(self.dtn3LayersWithTTable)

            self.transitionTable.end_run(True)

            toSaveDtn = self.dqn.end_run(r,sumR,numSteps)
            self.dtn3Layers.end_run(toSaveDtn)
            self.dtn4Layers.end_run(toSaveDtn)
            self.dtn5Layers.end_run(toSaveDtn)
            self.dtn3LayersWithTTable.end_run(toSaveDtn)
            
    def TestTransition(self, numTests, num2Print, fTest):
        mseResults = [0.0, 0.0, 0.0, 0.0, 0.0]

        for i in range(numTests):
            s = self.env.randomState()
            validActions = self.env.ValidActions(s)
            a = np.random.choice(validActions)
            
            outDtn3Layers = self.dtn3Layers.predict(s,a)
            outDtn4Layers = self.dtn4Layers.predict(s,a)
            outDtn5Layers = self.dtn5Layers.predict(s,a)
            outDtn3LayersTT = self.dtn3LayersWithTTable.predict(s,a)
            outTransitionTable = self.CalcDistTTable(s,a)                        

            realDistribution = self.env.RealDistribution(s,a)
         
            mseDTN3L = sum(pow(outDtn3Layers[0] - realDistribution, 2))
            mseDTN3LTT = sum(pow(outDtn3LayersTT[0] - realDistribution, 2))
            mseDTN4L = sum(pow(outDtn4Layers[0] - realDistribution, 2))
            mseDTN5L = sum(pow(outDtn5Layers[0] - realDistribution, 2))
            mseTable = sum(pow(outTransitionTable - realDistribution, 2))
            
            mseResults[0] += mseDTN3L / numTests
            mseResults[1] += mseDTN4L / numTests
            mseResults[2] += mseDTN5L / numTests
            mseResults[3] += mseDTN3LTT / numTests
            mseResults[4] += mseTable / numTests

            if i < num2Print:
                string4File = "for state = " + str(s) + " action = " + str(a) + ":\n\n"
                string4File += "\ndtn3Layer = \n" + str(outDtn3Layers) + "\n"
                string4File += "\ndtn4Layer  = \n" + str(outDtn4Layers) + "\n"
                string4File += "\ndtn5Layer  = \n" + str(outDtn5Layers) + "\n"
                string4File += "\ndtn3Layer With TTable = \n" + str(outDtn3LayersTT) + "\n"
                string4File += "\ntable = \n" + str(outTransitionTable) + "\n\n\n\n"

                fTest.write(string4File)

        
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

    def TrainAccording2TTable(self, dtn):
        states = list(self.transitionTable.table.keys())
        
        sLearn = []
        aLearn = []
        s_Learn = []

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

    def EndRound(self):
        return
        self.dtnWithTTable.NewTTable()


if __name__ == "__main__":
    
    dirName = "maze_game"
    fTestResultsName = "./" + dirName + "/maze_transition_results.txt"
    fMseResultsName = "./" + dirName + "/maze_mse_results.txt"
    fMsePlotName = "./" + dirName + "/maze_mse_results.png"

    numRounds = 500
    numTrialsInRound = 20
    rounds2DeleteTTable = 5

    sim = Simulator(dirName=dirName, trials2Save=numTrialsInRound)
    leg = ["dtn_with_3Layers", "dtn_with_4Layers", "dtn_with_5Layers", "dtn_with_3Layers with ttable", 'ttable']
    mseResults = []

    fTestResults = open(fTestResultsName, "w+")
    fMseResults = open(fMseResultsName, "w+")

    for i in range(numRounds):
        header = "\n\n\nresults before round " + str(i) + "(of " + str(numTrialsInRound) + " trials):\n"
        fTestResults.write(header)
        fMseResults.write(header)
        mse = sim.TestTransition(500, 5, fTestResults)
        fMseResults.write(str(mse) + "\n\n")
        mseResults.append(mse)

        fTestResults.close()
        fTestResults = open(fTestResultsName, "a+")
        fMseResults.close()
        fMseResults = open(fMseResultsName, "a+")

        fig = plt.figure()
        plt.plot(mseResults)
        plt.legend(leg)
        plt.title("mse results for maze")
        fig.savefig(fMsePlotName)

        sim.Simulate(numTrialsInRound)
        if i % rounds2DeleteTTable == rounds2DeleteTTable - 1:
            sim.EndRound()






