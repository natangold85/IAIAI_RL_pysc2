import numpy as np 
import random

from utils_tables import LearnWithReplayMngr
from utils_tables import DQN_EMBEDDING_PARAMS

TYPE = "type"
NN = "nn"
Q_TABLE = "q"
T_TABLE = "t"
HISTORY = "hist"
R_TABLE = "r"
RESULTS = "results"
PARAMS = 'params'
GRIDSIZE_key = 'gridsize'

RUN_TYPES = {}

RUN_TYPES['master'] = {}
RUN_TYPES['master'][TYPE] = "NN"
RUN_TYPES['master'][GRIDSIZE_key] = 5
RUN_TYPES['master'][PARAMS] = DQN_EMBEDDING_PARAMS(76, 75, 3)
RUN_TYPES['master'][NN] = "battleMngr_dqnGS5_Embedding_DQN"
RUN_TYPES['master'][Q_TABLE] = ""
RUN_TYPES['master'][HISTORY] = "battleMngr_dqnGS5_Embedding_replayHistory"
RUN_TYPES['master'][RESULTS] = "battleMngr_dqnGS5_Embedding_result"

RUN_TYPES['sub1'] = {}
RUN_TYPES['sub1'][TYPE] = "NN"
RUN_TYPES['sub1'][GRIDSIZE_key] = 5
RUN_TYPES['sub1'][PARAMS] = DQN_EMBEDDING_PARAMS(51, 50, 26)
RUN_TYPES['sub1'][NN] = "armyBattle_dqnGS5_Embedding_DQN"
RUN_TYPES['sub1'][Q_TABLE] = ""
RUN_TYPES['sub1'][HISTORY] = "armyBattle_dqnGS5_Embedding_replayHistory"
RUN_TYPES['sub1'][RESULTS] = "armyBattle_dqnGS5_Embedding_result"

RUN_TYPES['sub2'] = {}
RUN_TYPES['sub2'][TYPE] = "NN"
RUN_TYPES['sub2'][GRIDSIZE_key] = 5
RUN_TYPES['sub2'][PARAMS] = DQN_EMBEDDING_PARAMS(51, 50, 26)
RUN_TYPES['sub2'][NN] = "baseBattle_dqnGS5_Embedding_DQN"
RUN_TYPES['sub2'][Q_TABLE] = ""
RUN_TYPES['sub2'][HISTORY] = "baseBattle_dqnGS5_Embedding_replayHistory"
RUN_TYPES['sub2'][RESULTS] = "baseBattle_dqnGS5_Embedding_result"

class Sub:
    def __init__(self, runType):
        
        self.dqn = LearnWithReplayMngr(runType[TYPE], runType[PARAMS], runType[NN], runType[Q_TABLE], runType[RESULTS], runType[HISTORY])
        self.stateSize = runType[PARAMS].stateSize
        self.numActions = runType[PARAMS].numActions

    def Learn(self):
        for i in range(500):
            t = False
            steps = 0
            while not t:
                s,a,r,s_,t = self.CreateSample()
                self.dqn.learn(s,a,r,s_,t)
                steps += 1
            self.dqn.end_run(0,0,steps)
    
    def CreateSample(self):
        sBase = np.random.rand(self.stateSize)
        s_Base = np.random.rand(self.stateSize)
        
        s = np.zeros(self.stateSize, dtype = int)
        s_ = np.zeros(self.stateSize, dtype = int)
        for i in range(self.stateSize):
            s[i] = int(sBase[i] * 10)
            s_[i] = int(s_Base[i] * 10)        

        a = random.randint(0, self.numActions - 1)

        r = np.random.uniform()

        if s_[0] == 0:
            t = True
        else:
            t = False
    
        return s, a, r, s_, t

    def saveNN(self):
        self.dqn.dqn.save_network()
        



class Master(Sub):
    def __init__(self):
        super(Master, self).__init__(RUN_TYPES['master'])

        self.s1 = Sub(RUN_TYPES["sub1"])
        self.s2 = Sub(RUN_TYPES["sub2"])
    
    def Learn(self, n = 0):
        if n == 0:
            super(Master, self).Learn()
        elif n == 1:
            self.s1.Learn()
        elif n == 2:
            self.s2.Learn()         

    def saveNN(self, n = 0):
        if n == 0:
            super(Master, self).saveNN()
        elif n == 1:
            self.s1.saveNN()
        elif n == 2:
            self.s2.saveNN() 

t = Master()
t.Learn(0)


