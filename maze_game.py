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

class MazeGame(Maze):
    def __init__(self, gridSize = 5, holesCoord = None):
        super(MazeGame, self).__init__(gridSize = gridSize)
        self.stateSize = self.gridSize * self.gridSize + 1

        self.action_outFromPenalty = 5
        self.numActions = 6

        self.numSteps = 0
        self.maxSteps = 1000

        self.startingPntIdx = 0
        self.targetIdx = self.gridSize * self.gridSize - 1

        self.reward_InPenaltyLoc = -0.25
        self.reward_IlligalMove = -0.1

        if holesCoord == None:
            holesCoord = []
            holesCoord.append([3,2])
            holesCoord.append([1,3])
            holesCoord.append([2,2])
            
        self.holesIdx = []
        for coord in holesCoord:
            self.holesIdx.append(self.coord2Idx(coord))
 
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