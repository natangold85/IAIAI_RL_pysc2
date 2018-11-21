import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys
import threading
import math

import tensorflow as tf

from multiprocessing import Lock

from utils import EmptyLock

#decision makers
from algo_dqn import DQN
from algo_dqn import DQN_WithTarget
from algo_dqn import DQN_WithTargetAndDefault

from algo_a2c import A2C
from algo_a3c import A3C

from algo_qtable import QLearningTable

from utils_history import HistoryMngr

# model builders:
from utils_ttable import TransitionTable

# results handlers
from utils_results import ResultFile

class BaseDecisionMaker:
    def __init__(self, agentName):
        self.trainFlag = False
        self.switchFlag = False

        self.resultFile = None
        self.secondResultFile = None
        self.decisionMaker = None
        self.historyMngr = None

        self.subAgentsDecisionMakers = {}
        self.switchCount = {}
        self.agentName = agentName

        self.copyTargetLock = Lock()
        self.CopyTarget2ModelNumRuns = -1

    def SetSubAgentDecisionMaker(self, key, decisionMaker):
        self.subAgentsDecisionMakers[key] = decisionMaker
        self.switchCount[key] = 0

    def GetSubAgentDecisionMaker(self, key):
        if key in self.subAgentsDecisionMakers.keys():
            return self.subAgentsDecisionMakers[key]
        else:
            return None

    def GetDecisionMakerByName(self, name):
        if self.agentName == name:
            return self
        else:
            for saDM in self.subAgentsDecisionMakers.values():
                if saDM != None:
                    dm = saDM.GetDecisionMakerByName(name)
                    if dm != None:
                        return dm
                
            return None
            
    def TrainAll(self):
        numTrial2Learn = -1
    
        for subDM in self.subAgentsDecisionMakers.values():
            if subDM != None:
                numTrial2LearnSa = subDM.TrainAll()
                if numTrial2LearnSa >= 0:
                    numTrial2Learn = numTrial2LearnSa
        
        return numTrial2Learn
        
    def InitModel(self, sess, resetModel):
        if self.decisionMaker != None:
            self.decisionMaker.InitModel(sess, resetModel)

        if self.resultFile != None and not resetModel:
            self.resultFile.Load()

        if self.secondResultFile != None and not resetModel:
            self.secondResultFile.Load()

        if self.historyMngr != None and not resetModel:
            self.historyMngr.Load()

        for subAgentDm in self.subAgentsDecisionMakers.values():
            if subAgentDm != None:
                subAgentDm.InitModel(sess, resetModel)

    def AddSwitch(self, idx, numSwitch, name, resultFile):
        if resultFile != None:
            if idx not in self.switchCount:
                self.switchCount[idx] = 0

            if self.switchCount[idx] <= numSwitch:
                self.switchCount[idx] = numSwitch + 1
                slotName = name + "_" + str(numSwitch)
                resultFile.AddSlot(slotName)

    def AddSlot(self, slotName):
        if self.resultFile != None:
            self.resultFile.AddSlot(slotName)

    def GoToNextResultFile(self, numFile):
        self.resultFile.GoToNextFile(numFile)

    def AddResultFile(self, resultFile):
        self.secondResultFile = resultFile
        
    def GetResultFile(self):
        return self.secondResultFile

    def Train(self):
        pass     

    def NumRuns(self):
        pass

    def choose_action(self, state, validActions, targetValues=False):
        pass

    def learn(self, s, a, r, s_, terminal = False):
        pass

    def ActionsValues(self,state, validActions, targetValues = False):
        pass

    def end_run(self, r, score = 0 ,steps = 0):
        pass

    def TrimHistory(self):
        pass
    
    def AddHistory(self):
        return None
    
    def DrawStateFromHist(self, realState=True):
        return None
    
    def TakeTargetDM(self, numRunsEnd):
        self.copyTargetLock.acquire()
        self.CopyTarget2ModelNumRuns = numRunsEnd    
        self.copyTargetLock.release()

    def CopyTarget2Model(self, numRuns):
        pass

    def GetMinReward(self):
        return 0.0
    
    def SetMinReward(self, r):
        pass

    def GetMaxReward(self):
        return 0.01
    
    def SetMaxReward(self, r):
        pass

    def DecisionMakerType(self):
        return "Base"

    def NumDfltRuns(self):
        return 0

class BaseNaiveDecisionMaker(BaseDecisionMaker):
    def __init__(self, numTrials2Save, agentName = "", resultFName = None, directory = None):
        super(BaseNaiveDecisionMaker, self).__init__(agentName)
        self.resultFName = resultFName
        self.trialNum = 0
        self.numTrials2Save = numTrials2Save

        if resultFName != None:
            self.lock = Lock()
            if directory != None:
                fullDirectoryName = "./" + directory +"/"
                if not os.path.isdir(fullDirectoryName):
                    os.makedirs(fullDirectoryName)
            else:
                fullDirectoryName = "./"

            self.resultFile = ResultFile(fullDirectoryName + resultFName, numTrials2Save, agentName)



    def end_run(self, r, score, steps):
        saveFile = False
        self.trialNum += 1
        
        print("\t", threading.current_thread().getName(), ":", self.agentName," #trials =", self.trialNum, "reward =", r)
        if self.resultFName != None:
            self.lock.acquire()
            if self.trialNum % self.numTrials2Save == 0:
                saveFile = True

            self.resultFile.end_run(r, score, steps, saveFile)
            self.lock.release()
       
        return saveFile 


class UserPlay(BaseDecisionMaker):
    def __init__(self, agentName = "", playWithInput = True, numActions = 1, actionDoNothing = 0):
        super(UserPlay, self).__init__(agentName)
        self.playWithInput = playWithInput
        self.numActions = numActions
        self.actionDoNothing = actionDoNothing

    def choose_action(self, state, validActions, targetValues=False):
        if self.playWithInput:
            a = input("insert action: ")
        else:
            a = self.actionDoNothing
        return int(a)

    def learn(self, s, a, r, s_, terminal = False):
        return None

    def ActionsValues(self,state, validActions, targetValues = False):
        return np.zeros(self.numActions,dtype = float)

    def end_run(self, r, score = 0 ,steps = 0):
        return False



class DecisionMakerAlgoBase(BaseDecisionMaker):
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', isMultiThreaded=False):
        super(DecisionMakerAlgoBase, self).__init__(agentName)

        # params
        self.params = modelParams
        
        # count of trials that model is active and not learning to trim history
        self.nonTrainingHistCount = 0
        # trial to DQN to learn(insert to model as learn trial num)
        self.trial2LearnModel = -1
        # sync mechanism
        self.endRunLock = Lock() if isMultiThreaded else EmptyLock()

        # create directory        
        if directory != "":
            fullDirectoryName = "./" + directory +"/"
            if not os.path.isdir(fullDirectoryName):
                os.makedirs(fullDirectoryName)
        else:
            fullDirectoryName = "./"


        # create result file
        if resultFileName != '':
            self.resultFile = ResultFile(fullDirectoryName + resultFileName, modelParams.numTrials2Save, self.agentName)
        
        # create history mngr class
        self.historyMngr = HistoryMngr(modelParams, historyFileName, fullDirectoryName, isMultiThreaded)
    	
        self.startScope = fullDirectoryName
        # create decision maker class
        decisionClass = eval(modelType)
        with tf.variable_scope(self.startScope):
            self.decisionMaker = decisionClass(modelParams, decisionMakerName, fullDirectoryName, isMultiThreaded=isMultiThreaded, agentName=self.agentName)
        
    def AddHistory(self):
        return self.historyMngr.AddHistory()

    def choose_action(self, state, validActions, targetValues=False):
        if not self.decisionMaker.TakeDfltValues() and self.params.normalizeState:
            state = self.historyMngr.NormalizeState(state)

        return self.decisionMaker.choose_action(state, validActions, targetValues)     

    def NumRuns(self):
        return self.decisionMaker.NumRuns()

    def TrimHistory(self):
        count = self.nonTrainingHistCount + 1
        if count % self.params.numTrials2Learn == 0:
            self.historyMngr.TrimHistory()
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->Trim History to size =", self.historyMngr.Size())

        self.nonTrainingHistCount += 1

    def ResetHistory(self, dump2Old=True, save=False):
        self.historyMngr.Reset(dump2Old, save)

    def ResetAllData(self, resetDecisionMaker=True, resetHistory=True, resetResults=True):
        if resetDecisionMaker:
            self.decisionMaker.Reset()

        if resetHistory:
            self.historyMngr.Reset()

        if resetResults and self.resultFile != None:
            self.resultFile.Reset()


    def ActionsValues(self, state, validActions, targetValues = False):
        if not self.decisionMaker.TakeDfltValues():
            state = self.historyMngr.NormalizeState(state)

        return self.decisionMaker.ActionsValues(state, validActions, targetValues)

    def CopyTarget2Model(self, numRuns):
        print("\t", threading.current_thread().getName(), ":", self.agentName,"->Copy Target 2 Model")
        self.decisionMaker.CopyTarget2Model(numRuns)
    
    def DiscountFactor(self):
        return self.decisionMaker.DiscountFactor()

    def DrawStateFromHist(self, realState=True):
        return self.historyMngr.DrawState(realState)

    def GetMinReward(self):
        return self.historyMngr.GetMinReward()
    
    def SetMinReward(self, r):
        self.historyMngr.SetMinReward(r)

    def GetMaxReward(self):
        return self.historyMngr.GetMaxReward()
    
    def SetMaxReward(self, r):
        self.historyMngr.SetMaxReward(r)

    def DecisionMakerType(self):
        return self.decisionMaker.DecisionMakerType()

    def NumDfltRuns(self):
        return self.decisionMaker.NumDfltRuns()

    def DfltValueInitialized(self):
        return self.decisionMaker.DfltValueInitialized()

    def CheckModel(self, plotGraphs):
        pass

    def Save(self):
        self.decisionMaker.Save()
        self.historyMngr.Save()
        if self.resultFile != None:
            self.resultFile.Save()

    def end_test_run(self, r, score, steps):
        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, True)

    def NumTestRuns(self):
        numRuns = 0
        if self.resultFile != None:
            numRuns = self.resultFile.NumRuns()
        
        return numRuns
        
        


  
class DecisionMakerExperienceReplay(DecisionMakerAlgoBase):
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', isMultiThreaded = False):
        super(DecisionMakerExperienceReplay, self).__init__( modelType=modelType, modelParams=modelParams, agentName=agentName, decisionMakerName=decisionMakerName, 
                                                            resultFileName=resultFileName, historyFileName=historyFileName, directory=directory, 
                                                            isMultiThreaded=isMultiThreaded)

    def end_run(self, r, score, steps):
        self.endRunLock.acquire()

        numRun = int(self.NumRuns())
        #print(threading.current_thread().getName(), ":", self.agentName,"->for trial#", numRun, ": reward =", r, "score =", score, "steps =", steps)

        save = True if (numRun + 1) % self.params.numTrials2Save == 0 else False
        train = True if (numRun + 1) % self.params.numTrials2Learn == 0 else False

        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, save)

        if train:
            self.trial2LearnModel = numRun + 1
            self.trainFlag = True 
                
        self.decisionMaker.end_run(r)

        self.endRunLock.release()
        
        return save 

    def TrainAll(self):
        self.copyTargetLock.acquire()
        if self.CopyTarget2ModelNumRuns > 0:
            self.CopyTarget2Model(self.CopyTarget2ModelNumRuns)
            self.CopyTarget2ModelNumRuns = -1
        self.copyTargetLock.release()


        numTrial2Learn = -1
        
        if self.trainFlag:
            numTrial2Learn = self.Train()
            self.trainFlag = False

        numTrialsSa = super(DecisionMakerExperienceReplay, self).TrainAll()
       
        return numTrial2Learn if numTrial2Learn >= 0 else numTrialsSa

    def Train(self):
        s,a,r,s_, terminal = self.historyMngr.GetHistory()
        numRuns2Learn = -1
       
        if len(a) > self.params.minReplaySize:
            start = datetime.datetime.now()
            
            self.decisionMaker.learn(s, a, r, s_, terminal, self.trial2LearnModel)
            self.endRunLock.acquire()
            numRuns2Learn = self.trial2LearnModel
            self.decisionMaker.Save(self.trial2LearnModel)
            self.endRunLock.release()

            diff = datetime.datetime.now() - start
            msDiff = diff.seconds * 1000 + diff.microseconds / 1000
            
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->ExperienceReplay - training with hist size = ", len(r), ", last", msDiff, "milliseconds")
        else:
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->ExperienceReplay size to small - training with hist size = ", len(r))

        return numRuns2Learn

    def ActionsValues(self, state, validActions, targetValues = False):
        if not self.decisionMaker.TakeDfltValues():
            state = self.historyMngr.NormalizeState(state)

        return self.decisionMaker.ActionsValues(state, targetValues)

    def CheckModel(self, agent, plotGraphs=False, withDfltModel=False, statesIdx2Check=[], actions2Check=[]):
        
        np.set_printoptions(precision=2, suppress=True)
        
        print(self.agentName, "hist size =", len(self.historyMngr.transitions["a"]))
        print(self.agentName, "old hist size", len(self.historyMngr.oldTransitions["a"]))
        print(self.agentName, "all history size", len(self.historyMngr.GetAllHist()["a"]))

        print(self.agentName, "maxVals =", self.historyMngr.transitions["maxStateVals"])
        print(self.agentName, "maxReward =", self.historyMngr.GetMaxReward(), "minReward =", self.historyMngr.GetMinReward())
        print("\n")

        numStates2Check = 100

        states2Check = []
        vals = []
        valsTarget = []
        for i in range(numStates2Check):
            s = self.DrawStateFromHist()
            if len(s) > 0:
                states2Check.append(s)
                validActions = agent.ValidActions4State(s)
                vals.append(self.ActionsValues(s, validActions, targetValues=False))
                valsTarget.append(self.ActionsValues(s, validActions, targetValues=True))

        print(self.agentName, ": current dqn num runs = ", self.decisionMaker.NumRuns()," avg values =", np.average(vals, axis=0))
        print(self.agentName, ": target dqn num runs = ", self.decisionMaker.NumRunsTarget()," avg values =", np.average(valsTarget, axis=0))
        
        if withDfltModel:
            print("dqn value =", self.decisionMaker.ValueDqn(), "target value =", self.decisionMaker.ValueTarget(), "heuristic values =", self.decisionMaker.ValueDefault())
        else:
            print("dqn value =", self.decisionMaker.ValueDqn(), "target value =", self.decisionMaker.ValueTarget())

        print("\n\n")

        if plotGraphs:
            self.decisionMaker.CreateModelGraphs(agent, statesIdx=statesIdx2Check, actions2Check=actions2Check)
            self.decisionMaker.CreateModelGraphs(agent, statesIdx=statesIdx2Check, actions2Check=actions2Check, plotTarget=True)

    def CreateModelGraphs(self, agent, stateIdx2Check, actions2Check, plotTarget=False, dir2Save=None, numTrials=-1, maxSize2Plot=20000):
        import matplotlib.pyplot as plt
        from utils_plot import plotImg

        plotType = "target" if plotTarget else "current"

        figVals = None
        isFigVals = False
        figDiff = None
        isFigDiff = False

        idxX = stateIdx2Check[0]
        idxY = stateIdx2Check[1]
        
        if plotTarget:
            numRuns = self.decisionMaker.NumRunsTarget()
        else:
            numRuns = numTrials if numTrials >= 0 else self.decisionMaker.NumRuns()

        xName = agent.StateIdx2Str(idxX)
        yName = agent.StateIdx2Str(idxY)

        actionsPoints = {}

        # extracting nn vals for current nn and target nn

        for a in actions2Check:
            actionsPoints[a] = [[], [], [], []]

        sizeHist = len(self.historyMngr.Size())
        size2Plot = min(sizeHist, maxSize2Plot)
        for i in range(size2Plot):
            s = self.DrawStateFromHist(realState=False)
            validActions = agent.ValidActions4State(s)
            vals = self.ActionsValues(s, validActions, targetValues=plotTarget)
            
            if xName == "min" or xName == "MIN":
                s[idxX] = int(s[idxX] / 25) 
            if yName == "min" or yName == "MIN":
                s[idxY] = int(s[idxY] / 25) 

            def addPoint(x, y, val, actionVec):
                for i in range(len(actionVec[0])):
                    if x == actionVec[0][i] and y == actionVec[1][i]:
                        actionVec[2][i].append(val)
                        return
                
                actionVec[0].append(x)
                actionVec[1].append(y)
                actionVec[2].append([val])
                actionVec[3].append(0)


            for a in actions2Check:
                if a in validActions:
                    addPoint(s[idxX], s[idxY], vals[a], actionsPoints[a])
                else:
                    addPoint(s[idxX], s[idxY], np.nan, actionsPoints[a])

        # calculating avg val
        maxVal = -1.0
        minVal = 1.0

        for a in actions2Check:
            for i in range(len(actionsPoints[a][0])):
                actionsPoints[a][3][i] = np.nanmean(np.array(actionsPoints[a][2][i])) 
                maxVal = max(maxVal, actionsPoints[a][3][i])
                minVal = min(minVal, actionsPoints[a][3][i])

        
        numRows = math.ceil(len(actions2Check) / 2)
        idxPlot = 1

        figVals = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle("action evaluation - " + plotType + ": (#trials = " + str(numRuns) + ")")
        for a in actions2Check:
            x = np.array(actionsPoints[a][0])
            y = np.array(actionsPoints[a][1])
            z = np.array(actionsPoints[a][3])
            ax = figVals.add_subplot(numRows, 2, idxPlot)
            img = plotImg(ax, x, y, z, xName, yName, "values for action = " + agent.Action2Str(a, onlyAgent=True), minZ=minVal, maxZ=maxVal)
            if img != None:
                isFigVals = True
                figVals.colorbar(img, shrink=0.4, aspect=5)
                idxPlot += 1
        
        idxPlot = 1

        numRows = math.ceil(len(actions2Check) * (len(actions2Check) - 1) / 2)

        figDiff = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle("differrence in action values - " + plotType + ": (#trials = " + str(numRuns) + ")")
        idxPlot = 1

        for a1Idx in range(len(actions2Check)):
            a1 = actions2Check[a1Idx]
            x = np.array(actionsPoints[a1][0])
            y = np.array(actionsPoints[a1][1])
            z1 = np.array(actionsPoints[a1][3])

            if len(z1) == 0:
                continue

            for a2Idx in range(a1Idx + 1, len(actions2Check)):
                a2 = actions2Check[a2Idx]
                z2 = np.array(actionsPoints[a2][3])

                zDiff = z1 - z2
                maxZ = np.max(np.abs(zDiff))
                ax = figDiff.add_subplot(numRows, 2, idxPlot)
                title = "values for differrence = " + agent.Action2Str(a1, onlyAgent=True) + " - " + agent.Action2Str(a2, onlyAgent=True)
                img = plotImg(ax, x, y, zDiff, xName, yName, title, minZ=-maxZ, maxZ=maxZ)
                if img != None:
                    isFigDiff = True
                    figDiff.colorbar(img, shrink=0.4, aspect=5)
                    idxPlot += 1
    
        if dir2Save != None:
            if isFigVals:
                figVals.savefig(dir2Save + plotType + "DQN_" + str(numRuns))
            if isFigDiff:
                figDiff.savefig(dir2Save + plotType + "DQNDiff_" + str(numRuns))



class DecisionMakerOnlineAsync(DecisionMakerAlgoBase):
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', isMultiThreaded = False):        
        super(DecisionMakerOnlineAsync, self).__init__( modelType=modelType, modelParams=modelParams, agentName=agentName, decisionMakerName=decisionMakerName, 
                                                            resultFileName=resultFileName, historyFileName=historyFileName, directory=directory, 
                                                            isMultiThreaded=isMultiThreaded)

        self.agentsHistory = {}
        
        self.counter4HistFiles = 0
        self.histFileName = directory + "/histTest_"
        self.histFileLock = Lock()

    def AddHistory(self):
        idxWorker = len(self.agentsHistory)
        agentName = "GameThread_" + str(idxWorker)

        with tf.variable_scope(self.startScope):
            self.decisionMaker.AddWorker(agentName)

        hist = self.historyMngr.AddHistory()
        self.agentsHistory[agentName] = hist
       
        return hist

    def NumRunsAgent(self, agentName):
        return self.decisionMaker.NumRunsAgent(agentName)

    def NumRuns(self):
        runs = 0
        for name in self.agentsHistory.keys():
            runs += self.NumRunsAgent(name)

        return runs

    def end_run(self, r, score, steps):

        agentName = threading.current_thread().getName()
        numAgentRuns = self.NumRunsAgent(agentName)
        numTotRun = self.NumRuns()

        print(agentName, ":", self.agentName,"->for trial#", numTotRun, " agent trial#", numAgentRuns, ": reward =", r, "score =", score, "steps =", steps)

        self.decisionMaker.end_run(r)
        if (numAgentRuns + 1) % self.params.numTrials2Learn == 0:
            self.Train()            
             
        # for end run only shared files between agents are needed for lock
        self.endRunLock.acquire()

        save = True if (numTotRun + 1) % self.params.numTrials2Save == 0 else False
        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, save)
        if save:
            self.decisionMaker.Save()
            self.historyMngr.Save()

        self.endRunLock.release()

        return True

    def SaveLocalHistFile(self, s, a, r, s_, terminal):
        self.histFileLock.acquire()
        
        transitions = {}
        self.historyMngr.AddMetaDataFields2Dict(transitions)
        transitions["s"] = (s  + 1) * transitions["maxStateVals"] / 2
        transitions["a"] = a
        transitions["r"] = r
        transitions["s_"] = (s_  + 1) * transitions["maxStateVals"] / 2
        transitions["terminal"] = terminal
        transitions["worker"] = threading.current_thread().getName()

        fName = self.histFileName + str(self.counter4HistFiles)
        self.counter4HistFiles += 1

        self.histFileLock.release()

        pd.to_pickle(transitions, fName + '.gz', 'gzip') 

    def GetHistory(self, agentName):
        hist = self.agentsHistory[agentName]   
        s, a, r, s_, terminal = self.historyMngr.GetHistory(hist, shuffle=False)
        # if self.counter4HistFiles < 2000:
        #     self.SaveLocalHistFile(s, a, r, s_, terminal)

        return s, a, r, s_, terminal

    def Train(self):
        agentName = threading.current_thread().getName()

        s,a,r,s_, terminal = self.GetHistory(agentName)

        start = datetime.datetime.now()      
        self.decisionMaker.learn(s, a, r, s_, terminal, insert2Graph=True)
        diff = datetime.datetime.now() - start
        msDiff = diff.seconds * 1000 + diff.microseconds / 1000
        
        #print("\t", threading.current_thread().getName(), ":", self.agentName,"-> training with hist size = ", len(r), ", last", msDiff, "milliseconds")

        return self.decisionMaker.NumRuns()

    def CheckModel(self, agent, plotGraphs=False, withDfltModel=False, statesIdx2Check=[], actions2Check=[]):
        np.set_printoptions(precision=2, suppress=True)

        numStates2Check = 1000

        states2Check = []
        vals = []

        validActionsResults = {}
        for i in range(numStates2Check):
            s = self.DrawStateFromHist(realState=False)
            if len(s) > 0:
                states2Check.append(s)
                validActions = agent.ValidActions(s)
                v = self.ActionsValues(s, validActions, targetValues=True)
                
                keyValidActions = str(validActions)
                if keyValidActions not in validActionsResults:
                    validActionsResults[keyValidActions] = []

                validActionsResults[keyValidActions].append(v[validActions])

                vWithNan = np.zeros(len(v), float)
                vWithNan.fill(np.nan)
                for i in validActions:
                    vWithNan[i] = v[i]
                vals.append(vWithNan)
        print("average network state:")
        print(self.agentName, "maxVals =", self.historyMngr.transitions["maxStateVals"])
        print(self.agentName, ": current a3c num runs = ", self.decisionMaker.NumRuns()," avg values =", np.nanmean(vals, axis=0))

        print("\n\nnetwork state according to valid actions:")
        for key, value in validActionsResults.items():
            print("for valid actions:", key, "avg val =", np.average(value, axis=0))
