import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from utils_tables import DQN_PARAMS
from utils_dqn import LearnWithReplayMngr
from utils_dtn import DTN

class MazeGame:
    def __init__(self):
        self.gridSize = 5
        self.stateSize = self.gridSize * self.gridSize + 1

        self.startingPntIdx = 0
        self.targetIdx = 24

        self.rewardInPenaltyLoc = -0.25

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
        self.holesIdx.append(self.coord2Idx([4,4]))

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
        return s

    def InWormHole(self, loc):
        return loc in self.holesIdx

    def step(self, s, a):
        loc = (s == 1).nonzero()[0][0]
        if loc == self.targetIdx:
            return s, 1.0, True

        if s[self.penaltyIdx] == 1:
            if self.counterPenalty == 0:
                return s.copy(), -1.0, True
            
            r = self.rewardInPenaltyLoc
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

    
def CreateDQN(nnName, nnFunc = None, toTrain = 64, stateSize = 2, maxVal = 2):
    numActions = 2 * stateSize
    p = DQN_PARAMS(stateSize,numActions, explorationProb = 0.0, exploreChangeRate = 0.005, nn_Func = nnFunc)
    dqn = LearnWithReplayMngr('NN', p, nnName, numTrials2Learn = toTrain, newRun = True)
    return dqn

def Train(dqn, numRuns = 64, stateSize = 2, maxVal = 2):
    countWins = 0
    countLoss = 0
    for run in range(numRuns):
        s = np.zeros(stateSize, dtype= int)
        for n in range(stateSize):
            s[n] = random.randint(-maxVal,maxVal)
        
        result = True
        terminal = False
        numSteps = 0
        while not terminal:
            numSteps += 1
            a = dqn.choose_action(s)
            s_ = s.copy()
            if a < stateSize:
                s_[a] = s[a] + 1
            else:
                idx = a - stateSize
                s_[idx] = s[idx] - 1

            r = 1.0
            win = True
            loss = False
            for n in range(stateSize):
                if s_[n] != 0:
                    win = False
                    r = 0.0
                    if abs(s_[n]) > maxVal:
                        loss = True
                        result = False
                        r = -1.0
                        break
            
            terminal = win or loss
            dqn.learn(s,a,r,s_,terminal)
            s = s_.copy()

        dqn.end_run(r,0,numSteps)
        if result:
            countWins += 1
        else:
            countLoss += 1

    #allVars = tf.all_variables()

    return (countWins - countLoss) / numRuns
    