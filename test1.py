import numpy as np
import pandas as pd
import threading

import tensorflow as tf


from algo_dqn import DQN
from algo_a2c import A2C
from algo_a3c import A3C

from algo_dqn import DQN_PARAMS
from algo_a2c import A2C_PARAMS
from algo_a3c import A3C_PARAMS

GRID_SIZE = 5

STATE_START_SELF_MAT = 0
STATE_END_SELF_MAT = STATE_START_SELF_MAT + GRID_SIZE * GRID_SIZE
STATE_START_ENEMY_MAT = STATE_END_SELF_MAT
STATE_END_ENEMY_MAT = STATE_START_ENEMY_MAT + GRID_SIZE * GRID_SIZE
STATE_TIME_LINE_IDX = STATE_END_ENEMY_MAT
STATE_SIZE = STATE_TIME_LINE_IDX + 1

ACTION_DO_NOTHING = 0
ACTIONS_START_IDX_ATTACK = 1
ACTIONS_END_IDX_ATTACK = ACTIONS_START_IDX_ATTACK + GRID_SIZE * GRID_SIZE
NUM_ACTIONS = ACTIONS_END_IDX_ATTACK

NUM_TRIALS_2_SAVE = 100

def ValidActions(state):
    valid = [ACTION_DO_NOTHING]
    enemiesLoc = (state[STATE_START_ENEMY_MAT:STATE_END_ENEMY_MAT] > 0).nonzero()
    for loc in enemiesLoc[0]:
        valid.append(loc + ACTIONS_START_IDX_ATTACK)

    return valid    

# run from cpu
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

threadName = "test1T"
threading.current_thread().setName("test1T")

paramsDqn = DQN_PARAMS(STATE_SIZE, NUM_ACTIONS, layersNum=2, numTrials2Save=NUM_TRIALS_2_SAVE)

with tf.variable_scope("DQN"):
    dqn = DQN(paramsDqn, nnName = "dqn", nnDirectory = "test1/testDQN", isMultiThreaded=True, agentName = "army_attack")

paramsA2C = A2C_PARAMS(STATE_SIZE, NUM_ACTIONS, numTrials2Save=NUM_TRIALS_2_SAVE)
with tf.variable_scope("A2C"):
    a2c = A2C(paramsA2C, nnName = "dqn", nnDirectory = "test1/testA2C", isMultiThreaded=True, agentName = "army_attack")

paramsA3C = A3C_PARAMS(STATE_SIZE, NUM_ACTIONS, numTrials2Learn=1, numTrials2Save=NUM_TRIALS_2_SAVE)
with tf.variable_scope("A3C"):
    a3c = A3C(paramsA3C, nnName = "dqn", nnDirectory = "test1/testA3C", isMultiThreaded=True, agentName = "army_attack")
with tf.variable_scope("A3C"):
    a3c.AddWorker(threading.current_thread().getName())

directory = "./TrainArmyAttackA3C/ArmyAttack/armyAttack_A3C/"
testHistFName = "histTest_"

np.set_printoptions(precision=2, suppress=True)

with tf.Session() as sess:
    
    dqn.InitModel(session=sess, resetModel=True)
    a2c.InitModel(session=sess, resetModel=True)
    a3c.InitModel(session=sess, resetModel=True)

    for epoch in range(10):
        resultsA2C = []
        resultsA3C = []
        print("\n\n\nstart epoch #", epoch, "\n\n\n")
        for i in range(200):
            transitions = pd.read_pickle(directory + testHistFName + str(i) + '.gz', compression='gzip')
            s = transitions["s"]
            a = transitions["a"]
            r = transitions["r"]
            s_ = transitions["s_"]
            terminal = transitions["terminal"]

            sNorm = (s * 2) / transitions["maxStateVals"] - 1.0
            s_Norm = (s_ * 2) / transitions["maxStateVals"] - 1.0

            #dqn.learn(sNorm, a, r, s_Norm, terminal)
            a2c.learn(sNorm, a, r, s_Norm, terminal)
            a3c.learn(sNorm, a, r, s_Norm, terminal)

            if (i + 1) % 100 == 0:
                validActionsResultsA2C = {}
                validActionsResultsA3C = {}

                for idxS in range(len(s)):
                    validActions = ValidActions(s[idxS])
                    a2cVal = a2c.ActionsValues(sNorm[idxS], validActions)
                    a3cVal = a3c.ActionsValues(sNorm[idxS], validActions)

                    validActionsKey = str(validActions)
                    if validActionsKey not in validActionsResultsA2C:
                        validActionsResultsA2C[validActionsKey] = []
                        validActionsResultsA3C[validActionsKey] = []

                    validActionsResultsA2C[validActionsKey].append(a2cVal[validActions])
                    validActionsResultsA3C[validActionsKey].append(a3cVal[validActions])

                resultsA2C.append(validActionsResultsA2C)
                resultsA3C.append(validActionsResultsA3C)
                print("episode #", i + 1)
                for key in validActionsResultsA2C.keys():
                    print ("\nfor valid actions ", key)
                    print("\na2c val last=", np.average(validActionsResultsA2C[key], axis=0), "\na3c val last=", np.average(validActionsResultsA3C[key], axis=0))
                    if len(resultsA2C) > 1:
                        sumA2C = 0
                        sumA3C = 0
                        error = False
                        for idxResults in range(len(resultsA2C)):
                            if key in resultsA2C[idxResults]:
                                sumA2C += np.average(resultsA2C[idxResults][key], axis=0)
                            else:
                                error = True
                            if key in resultsA3C[idxResults]:
                                sumA3C += np.average(resultsA3C[idxResults][key], axis=0)
                            else:
                                error = True
                        
                        if not error:
                            print("\navg a2c val =", sumA2C / len(resultsA2C), "\navg a3c =", sumA3C / len(resultsA3C))

            a2c.Reset()
            a3c.Reset()
                        
            



