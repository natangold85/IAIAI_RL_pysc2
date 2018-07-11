import cProfile
import re

import numpy as np
import random
import matplotlib.pyplot as plt

import sys

from utils_tables import LearnWithReplayMngr
NN_FILE = "C:/Users/user/Documents/GitHub/pysc2_mcts_learning/nn_test"
NN_FILE = "./nn_test"
TABLE_FILE = "test_table"

class TEST:
    def __init__(self, sizeState, rangeOfValues = 1, valToEnd = 0, step2End = 20):
        self.numActions = sizeState * 2
        self.state = np.zeros(sizeState, dtype=int)
        self.env_name = "test"
        self.env_type = "1"

        self.minInitVal = valToEnd - rangeOfValues
        self.maxInitVal = valToEnd + rangeOfValues

        self.sizeState = sizeState
        self.valToEnd = valToEnd
        self.numStep = 0
        self.step2End = step2End

        self.winState = np.zeros(sizeState, dtype=int)

        self.lossState = np.zeros(sizeState, dtype=int)
        for i in range(sizeState):
            self.lossState[i] = - rangeOfValues - 1

        self.tieState = np.zeros(sizeState, dtype=int)
        for i in range(sizeState):
            self.tieState[i] = rangeOfValues + 1

    def Terminal(self, state):
        win = True
        loss = False

        for i in range(0, self.sizeState):
            if state[i] != self.valToEnd:
                win = False
            if state[i] > self.maxInitVal or state[i] < self.minInitVal:
                loss = True

        if loss:
            return True, -1.0
        elif win:
            return True, 1.0

        if self.numStep == self.step2End:
            return True, 0
        else:
            return False, 0

    def PrintState(self):
        ret = ""
        for i in range(self.sizeState):
            ret += str(self.state[i]) + ','

        return ret

    def new_random_game(self):
        self.numStep = 0
        for i in range(0, self.sizeState):
            self.state[i] = random.randint(self.minInitVal, self.maxInitVal)

        terminal, r = self.Terminal(self.state)
        if (terminal):
            return self.new_random_game()

        return self.state.copy()

    def new_game(self, state):
        self.state = state.copy()
        return state

    def act(self, action):
        self.numStep += 1
        if action < self.sizeState:
            self.state[action] += 1
        else:
            action -= self.sizeState
            self.state[action] -= 1

        terminal, r = self.Terminal(self.state)
        if not terminal:
            return self.state.copy(), r, terminal
        else:
            if r > 0: 
                return self.winState, r, terminal
            else:
                return self.lossState, r, terminal


    def GetAllPossibleStates(self):
        allStates = []
        state = np.zeros(self.sizeState, dtype=int)
        self.FillStatesRec(allStates, state)
        return allStates
    def GetTerminalStatesDict(self):
        stateDict = {}
        stateDict["win"] = self.winState
        stateDict["loss"] = self.lossState
        stateDict["tie"] = self.tieState

        return stateDict

    def FillStatesRec(self, allStates, state, currIdx = 0):
        if currIdx == self.sizeState:
            terminalState, _ = self.Terminal(state)
        if not terminalState:
            allStates.append(state.copy())
        else:
            for i in range(self.minInitVal, self.maxInitVal + 1):
                state[currIdx] = i
                self.FillStatesRec(allStates, state, currIdx + 1)


def Run(n = 200):
    sizeState = 2
    game = TEST(sizeState)
    terminalStates = game.GetTerminalStatesDict()

    if "qtable" in sys.argv:
        model = LearnWithReplayMngr(game.numActions, sizeState, False, terminalStates, "", TABLE_FILE)
    else:
        model = LearnWithReplayMngr(game.numActions, sizeState, True, terminalStates, NN_FILE)

    training_sessions = n
    gameEpochs = 500

    results = []
    resultsRandom = []
    sumReward = 0
    sumRewardRandom = 0
    for i in range(training_sessions):
        s = game.new_random_game()
        sRandom = game.new_game(s)
        terminalGame = False
        while not terminalGame:
            a = model.choose_action(s)
            s_, r, terminalGame = game.act(a)
            model.learn(s,a,r,s_)
            s = s_
        

        model.end_run(r)
        sumReward += r

        terminalRandom = False
        while not terminalRandom:
            a = random.randint(0, game.numActions - 1)
            sRandom, rRandom, terminalRandom = game.act(a)
        
        sumRewardRandom += rRandom
        if i % gameEpochs == gameEpochs - 1:
            results.append(sumReward / gameEpochs)
            sumReward = 0
            resultsRandom.append(sumRewardRandom / gameEpochs)
            sumRewardRandom = 0
        

    plt.plot(results)
    plt.plot(resultsRandom)
    plt.legend(["nn results", "random results"])
    #plt.show()


Run(3000)
# x = """Run(50)"""
# fileO = open("prof.csv", "w+")
# s = cProfile.run(x)
# fileO.write(str(s))
# print(s)




def callGrouping(size):
    x = []
    y = []
    for i in range(size):
        xi = random.randint(0,84)
        yi = random.randint(0,84)
        x.append(xi)
        y.append(yi)
    return Grouping(y,x)



import tensorflow as tf
import numpy as np
from utils_tables import DQN_PARAMS
from utils_dqn import DQN
import random

def build_dqn_0init(x, numActions, scope):
    with tf.variable_scope(scope):
        # Fully connected layers
        fc1 = tf.contrib.layers.fully_connected(x, 512, activation_fn = tf.nn.softplus, weights_initializer=tf.zeros_initializer())
        output = tf.contrib.layers.fully_connected(fc1, numActions, activation_fn = tf.nn.sigmoid, weights_initializer=tf.zeros_initializer()) * 2 - 1
    return output

p = DQN_PARAMS(2,4, nn_Func = build_dqn_0init)
dqn = DQN(p,'test', False)

size = 64
s = np.zeros(size, dtype= int)
s_ = np.zeros(size, dtype= int)
r = np.zeros(size, dtype= float)
a = np.zeros(size, dtype= int)
t = np.zeros(size, dtype= bool)

for i in range(64):
    s[i] = random.randint(-2,2)
    a[i] = random.randint(0,1)

for i in range(64):
    if a[i] == 0:
        s_[i] = s[i] + 1
    else:
        s_[i] = s[i] - 1

for i in range(64):
    if s_[i] == 0:
        t[i] = True
        r[i] = 1.0
    elif s_[i] > 2 or s_[i] < -2:
        t[i] = True
        r[i] = -1.0
    else:
        t[i] = False
        r[i] = 0.0

allVars = tf.all_variables()
dqn.learn(s,a,r,s_,t)
# from multiprocessing import Process
# import datetime
# def PSFunc():
#     s1 = 0
#     s2 = 0
#     s3 = 0
#     s4 = 0
#     d = {}
#     print("start")
#     for i in range(10000000):
#         s1 += i
#         if i % 2 == 0:
#             s2 += i
#         if i % 3 == 0:
#             s3 += i
#         if i % 4 == 0:
#             s4 += i
#         d[i] = s1 + s2 + s3 + s4
        
    
#     print(s1)

# if __name__ == '__main__':   
#     ps = []
#     dtStart = datetime.datetime.now()
#     for i in range(100):
#         ps.append(Process(target=PSFunc))
#         ps[i].start()

#     for i in range(100):
#         ps[i].join()


#     dtEnd = datetime.datetime.now()
#     print("seconds =", (dtEnd - dtStart).total_seconds())