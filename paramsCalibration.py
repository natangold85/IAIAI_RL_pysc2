import math
import threading
import os
import numpy as np
import tensorflow as tf

from algo_geneticProgramming import GP_Params
from algo_geneticProgramming import RationalParamType
from algo_geneticProgramming import Population
from algo_geneticProgramming import ParamsState

from utils_results import AvgResults
from utils_results import ChangeName2NextResultFile

from utils_history import GetHistoryFromFile
from utils_history import JoinTransitions

CURRENT_POPULATION_FILE_NAME = "gp_population.txt"
HISTORY_POPULATION_FILE_NAME = "gp_population_history.txt"

class SingleThreadData:
    def __init__(self, dirIdx, params):
        self.dirIdx = dirIdx
        self.params = params
        
class Calibration:
    def __init__(self, configDict, runType, paramsState, params, numInstances):
        self.numIndividualTraining = numInstances
        self.numThreadsTraining = 8

        self.configDict = configDict
        self.runType = runType

        self.paramsState = paramsState
        self.populationMngr = Population(paramsState, params)

        self.numGeneration = 0


    def Cycle(self, go2NextWOFitness=False):
        populationExist, fitnessExist = self.LoadPopulation()
        
        if populationExist and not go2NextWOFitness and not fitnessExist:
            return

        if populationExist:
            # go to next generation if prev generation exist
            self.populationMngr.Cycle()
            self.numGeneration += 1

        populationsAndDir = self.DividePopulation2Directory()
        self.SavePopulation(populationsAndDir)
            

    def DividePopulation2Directory(self):
        population = self.populationMngr.GetPopulation()
        pop2dir = []
        currD = 0
        for indiv in population:
            dire = list(np.arange(self.numIndividualTraining) + currD)
            pop2dir.append([dire, indiv])
            currD += self.numIndividualTraining
        
        return pop2dir
        
    def SavePopulation(self, populationsAndDir):
        populationDict = {}
        populationDict["numGeneration"] = self.numGeneration
        populationDict["population"] = populationsAndDir
        populationFName = self.configDict["directory"] + "/" + CURRENT_POPULATION_FILE_NAME

        open(populationFName, "w+").write(str(populationDict))

    def LoadPopulation(self):
        populationFName = self.configDict["directory"] + "/" + CURRENT_POPULATION_FILE_NAME
        if not os.path.isfile(populationFName):
            return False, False

        populationList = eval(open(populationFName, "r+").read())        
        self.numGeneration = populationList["numGeneration"]

        population = []
        for m in populationList["population"]:
            population.append(m[1])
        
        if "fitness" in populationList.keys():
            self.populationMngr.Population(population, np.array(populationList["fitness"]))
            return True, True
        else:
            self.populationMngr.Population(population)
            return True, False

    def Size(self):
        return self.populationMngr.Size()

def ChangeParamsAccordingToDict(params, paramsDict):
    if "learningRatePower" in paramsDict:
        params.learning_rateActor = 10 ** paramsDict["learningRatePower"]
        params.learning_rateCritic = 10 ** paramsDict["learningRatePower"]
    return params

def CreateParamsState(params2Calibrate):
    paramState = ParamsState()
    for pName in params2Calibrate:
        if "learningRatePower" == pName:
            paramState.AddParam(RationalParamType("learningRatePower", minVal=-9, maxVal=-2, breedProbType="normal"))

        elif "hundredsTrainEpisodes" == pName:
            paramState.AddParam(RationalParamType("hundredsTrainEpisodes", minVal=5, maxVal=50, floatingValueValid=False, breedProbType="normal"))

    return paramState

def SetPopulationTrained(directory):
    populationFName = directory + "/" + CURRENT_POPULATION_FILE_NAME
    if not os.path.isfile(populationFName):
        return False

    populationDict = eval(open(populationFName, "r+").read())   
    populationDict["trained"] = True

    open(populationFName, "w+").write(str(populationDict))
    
def GetPopulationDict(directory):
    populationFName = directory + "/" + CURRENT_POPULATION_FILE_NAME
    if not os.path.isfile(populationFName):
        return {}, 0

    populationDict = eval(open(populationFName, "r+").read())        
    return populationDict, populationDict["numGeneration"]

def DeletePopulation(directory):
    populationFName = directory + "/" + CURRENT_POPULATION_FILE_NAME
    if os.path.isfile(populationFName):
        os.remove(populationFName)

    populationHistoryFName = directory + "/" + HISTORY_POPULATION_FILE_NAME
    if os.path.isfile(populationHistoryFName):
        os.remove(populationHistoryFName)    


def GeneticProgrammingGeneration(populationSize, numInstances, configDict, runType): 
    parms2Calib = configDict["params2Calibrate"]
    paramsState = CreateParamsState(parms2Calib)

    params = GP_Params(populationSize=populationSize)
    
    calib = Calibration(configDict, runType, paramsState, params, numInstances)
    calib.Cycle(go2NextWOFitness=False)


def ReadGPFitness(configDict, agentName, runType): 
    populationFName = configDict["directory"] + "/" + CURRENT_POPULATION_FILE_NAME
    populationList = eval(open(populationFName, "r+").read()) 

    fitness = []

    for member in populationList["population"]:
        results = []
        for idx in member[0]:
            path = configDict["directory"] + "/" + agentName + "/" + runType["directory"]
            r = AvgResults(path, runType["results"], idx)
            ChangeName2NextResultFile(path, runType["results"], idx, populationList["numGeneration"])
            if r != None:
                results.append(r)

        if len(results) > 0:
            fitness.append(np.average(results))
        else:
            fitness.append(np.nan)

    populationList["fitness"] = fitness
    
    # sort according to fitness
    population = populationList["population"]
    population = [x for _,x in sorted(zip(fitness,population), reverse=True)]
    fitness = sorted(fitness, reverse=True)
    
    populationList["population"] = population
    populationList["fitness"] = fitness

    # save current population file
    open(populationFName, "w+").write(str(populationList))

    # save history population
    populationHistoryFName = configDict["directory"] + "/" + HISTORY_POPULATION_FILE_NAME
    
    if os.path.isfile(populationHistoryFName):
        populationHistory = eval(open(populationHistoryFName, "r+").read()) 
    else:
        populationHistory = []

    
    populationHistory.append(populationList)
    open(populationHistoryFName, "w+").write(str(populationHistory))



def TrainSingleGP(configDict, agentName, runType, dirCopyIdx):
    paramsDict = configDict["hyperParams"]
    
    if "hundredsTrainEpisodes" in paramsDict:
        numTrainEpisodes = paramsDict["hundredsTrainEpisodes"] * 100
    else:
        numTrainEpisodes = 5000

    transitions = ReadAllHistFile(configDict, agentName, runType, numTrainEpisodes)
    if transitions == {}:
        print("empty history return")
        return
    else:
        print("hist size read = ", np.sum(transitions["terminal"]), "num supposed to load =", numTrainEpisodes)
    
    from algo_decisionMaker import CreateDecisionMaker
    decisionMaker, _ = CreateDecisionMaker(agentName=agentName, configDict=configDict, isMultiThreaded=False, dmCopy=dirCopyIdx)

    with tf.Session() as sess:
        decisionMaker.InitModel(sess, resetModel=True)  
        s = transitions["s"]
        a = transitions["a"]
        r = transitions["r"]
        s_ = transitions["s_"]
        terminal = transitions["terminal"]

        numTrainEpisodes = paramsDict["hundredsTrainEpisodes"] * 100
        numTrain = 0
        
        training = True
        i = 0
        decisionMaker.ResetHistory(dump2Old=True, save=True)
        history = decisionMaker.AddHistory()
        decisionMaker.resultFile = None
        while training:
            history.learn(s[i], a[i], r[i], s_[i], terminal[i])
            
            if terminal[i]:
                decisionMaker.end_run(r[i], 0 ,0)
                if decisionMaker.trainFlag:
                    decisionMaker.Train()
                    decisionMaker.trainFlag = False
                
                numTrain += 1 
                if numTrain > numTrainEpisodes:
                    training = False
            
            i += 1
            if i == len(a):
                training = False
        
        decisionMaker.ResetHistory(dump2Old=False, save=False)

        print("end train --> num trains =", numTrain - 1, "num supposed =", numTrainEpisodes)
        decisionMaker.decisionMaker.Save()


def ReadAllHistFile(configDict, agentName, runType, numEpisodesLoad):
    maxHistFile = 200
    allTransitions = {}
        
    path = "./" + configDict["directory"] + "/" + agentName + "/" + runType["directory"]
    
    currNumEpisodes = 0
    idxHistFile = 0
    
    while numEpisodesLoad > currNumEpisodes and idxHistFile < maxHistFile:
        currFName = path + "_" + str(idxHistFile) + "/" + runType["history"] 

        if os.path.isfile(currFName + ".gz"):
            transitions = GetHistoryFromFile(currFName)
            if transitions != None:
                currNumEpisodes += np.sum(transitions["terminal"])
                JoinTransitions(allTransitions, transitions)
        
        idxHistFile += 1

    idxHistFile = 0
    while numEpisodesLoad > currNumEpisodes and idxHistFile < maxHistFile:
        currFName = path + "_" + str(idxHistFile) + "/" + runType["history"] + "_last"
        if os.path.isfile(currFName + ".gz"):
            transitions = GetHistoryFromFile(currFName)
            if transitions != None:
                currNumEpisodes += np.sum(transitions["terminal"])
                JoinTransitions(allTransitions, transitions)
        
        idxHistFile += 1
    
    return allTransitions


if __name__ == "__main__":
    import sys
    from absl import app
    from absl import flags
    import matplotlib.pyplot as plt

    from utils_plot import plotImg
    from utils_plot import PlotMeanWithInterval
    from agentRunTypes import GetRunType

    flags.DEFINE_string("directoryName", "none", "directory names to take results")
    flags.DEFINE_string("agentName", "none", "directory names to take results")
    flags.FLAGS(sys.argv)

    if "results" in sys.argv:
        configDict = eval(open("./" + flags.FLAGS.directoryName + "/config.txt", "r+").read())
        configDict["directory"] = flags.FLAGS.directoryName
        agentName = flags.FLAGS.agentName

        runType = GetRunType(agentName, configDict)

        popHistoryFname = "./" + flags.FLAGS.directoryName + "/" + HISTORY_POPULATION_FILE_NAME
        populationHistory = eval(open(popHistoryFname, "r+").read())

        params = configDict["params2Calibrate"]
        if len(params) != 2:
            print("ERROR: coordinate code to differrent param num")
            exit()

        paramsAll = [[], []]
        paramsAllAccording2Gen = [[], []]
        fitnessAll = []
        stdSinglePopulation = []

        generationPopulation = []
        genNum = len(populationHistory)

        numTopPopulation = 10
        
        fitnessGenAll = []
        fitnessGenTopPopulation = []
        fitnessGenBest = []

        for gen in range(0, genNum):
            genDict = populationHistory[gen]
            population = genDict["population"]
            fitness = np.array(genDict["fitness"])

            fitnessGenBest.append(np.max(fitness))
            fitnessGenAll.append(fitness)

            topPopulation = fitness[fitness.argsort()[-numTopPopulation:][::-1]]
            fitnessGenTopPopulation.append(topPopulation)

            params0 = []
            params1 = []
            for i in range(len(fitness)):
                params0.append(population[i][1][0])
                params1.append(population[i][1][1])
                fitnessAll.append(fitness[i])

            paramsAllAccording2Gen[0].append(params0)
            paramsAllAccording2Gen[1].append(params1)
            paramsAll[0] += params0
            paramsAll[1] += params1
            # load fitness from results file
            for i in range(len(population)):
                currFitnessDetailed  = []
                popInstances = population[i][0]
                for ins in popInstances:
                    path = "./" + configDict["directory"] + "/" + agentName + "/" +  runType["directory"]
                    r = AvgResults(path, runType["results"] + "_" + str(gen), ins)
                    if r != None:
                        currFitnessDetailed.append(r)
                
                stdSinglePopulation.append(np.std(currFitnessDetailed))



        figVals = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle("genetic programming results" )

        ax = figVals.add_subplot(3, 2, 1)
        
        PlotMeanWithInterval(np.arange(genNum), np.average(fitnessGenAll, axis=1), np.std(fitnessGenAll, axis=1), axes=ax)
        PlotMeanWithInterval(np.arange(genNum), np.average(fitnessGenTopPopulation, axis=1), np.std(fitnessGenTopPopulation, axis=1), axes=ax)
        ax.plot(np.arange(genNum), fitnessGenBest)

        ax.legend(["average population", "top " + str(numTopPopulation), "best"])
        ax.set_xlabel("generation num")
        ax.set_ylabel("fitness")
        ax.set_title("fitness results")

        ax = figVals.add_subplot(3, 2, 2)
        img = plotImg(ax, paramsAll[0], paramsAll[1], fitnessAll, params[0], params[1], "fitness value", binY=5, binX=1)
        figVals.colorbar(img, shrink=0.4, aspect=5)


        ax = figVals.add_subplot(3, 2, 3)
        ax.hist(stdSinglePopulation, bins=30)
        ax.set_title("histogram of std of single individual of population")
        ax.set_xlabel("std")

        ax = figVals.add_subplot(3, 2, 5)
        PlotMeanWithInterval(np.arange(genNum), np.average(paramsAllAccording2Gen[0], axis=1), np.std(paramsAllAccording2Gen[0], axis=1), axes=ax)
        ax.set_title("values of param =" + params[0])
        ax.set_xlabel("generation num")

        ax = figVals.add_subplot(3, 2, 6)
        PlotMeanWithInterval(np.arange(genNum), np.average(paramsAllAccording2Gen[1], axis=1), np.std(paramsAllAccording2Gen[1], axis=1), axes=ax)
        ax.set_title("values of param =" + params[1])
        ax.set_xlabel("generation num")

        plt.show()

