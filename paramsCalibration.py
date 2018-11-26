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
from utils_results import GoToNextResultFile

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
        
        if not go2NextWOFitness and not fitnessExist:
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
        return 0

    populationDict = eval(open(populationFName, "r+").read())        
    return populationDict, populationDict["numGeneration"]


def GeneticProgrammingGeneration(populationSize, numInstances, configDict, runType): 
    parms2Calib = configDict["params2Calibrate"]
    paramsState = CreateParamsState(parms2Calib)

    params = GP_Params(populationSize=populationSize)
    
    calib = Calibration(configDict, runType, paramsState, params, numInstances)
    calib.Cycle(go2NextWOFitness=False)


def ReadGPFitness(configDict, runType): 
    populationFName = configDict["directory"] + "/" + CURRENT_POPULATION_FILE_NAME
    populationList = eval(open(populationFName, "r+").read()) 

    fitness = []

    for member in populationList["population"]:
        results = []
        for idx in member[0]:
            # hard coded todo: do it better
            path = configDict["directory"] + "/ArmyAttack/" + runType["directory"]
            r = AvgResults(path, runType["results"], idx)
            GoToNextResultFile(path, runType["results"], idx, populationList["numGeneration"])
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



def TrainSingleGP(configDict, runType, dmInitFunc, dirCopyIdx):
    paramsDict = configDict["hyperParams"]
    
    if "hundredsTrainEpisodes" in paramsDict:
        numTrainEpisodes = paramsDict["hundredsTrainEpisodes"] * 100
    else:
        numTrainEpisodes = 5000

    transitions = ReadAllHistFile(configDict, runType, numTrainEpisodes)
    if transitions == {}:
        print("empty history return")
        return
    else:
        print("hist size read = ", np.sum(transitions["terminal"]), "num supposed to load =", numTrainEpisodes)

    decisionMaker, _ = dmInitFunc(configDict, isMultiThreaded=False, dmCopy=dirCopyIdx)

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


def ReadAllHistFile(configDict, runType, numEpisodesLoad):
    maxHistFile = 200
    allTransitions = {}
    
    # hard coded todo: fix it
    
    path = "./" + configDict["directory"] + "/ArmyAttack/" + runType["directory"]
    
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

    flags.DEFINE_string("directoryName", "", "directory names to take results")
    flags.FLAGS(sys.argv)

    if "results" in sys.argv:
        configDict = eval(open("./" + flags.FLAGS.directoryName + "/config.txt", "r+").read())
        configDict["directory"] = flags.FLAGS.directoryName

        from agent_army_attack import GetRunTypeArmyAttack
        ReadGPFitness(configDict, GetRunTypeArmyAttack(configDict))


        popHistoryFname = "./" + flags.FLAGS.directoryName + "/" + HISTORY_POPULATION_FILE_NAME
        populationHistory = eval(open(popHistoryFname, "r+").read())

        params = configDict["params2Calibrate"]
        if len(params) != 2:
            print("ERROR: coordinate code to differrent param num")
            exit()

        paramsAll = [[], []]
        fitnessAll = []

        generationPopulation = []
        genNum = len(populationHistory)

        avgFitnessGen = []
        avgFitnessGenTop3 = []
        avgFitnessGenBest = []
        for gen in range(0, genNum):
            genDict = populationHistory[gen]
            population = genDict["population"]
            fitness = np.array(genDict["fitness"])

            avgFitnessGen.append(np.average(fitness))
            avgFitnessGenBest.append(np.max(fitness))

            for i in range(len(fitness)):
                paramsAll[0].append(population[i][1][0])
                paramsAll[1].append(population[i][1][1])
                fitnessAll.append(fitness[i])

        figVals = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle("genetic programming results" )

        ax = figVals.add_subplot(2, 1, 1)
        ax.plot(np.arange(genNum), avgFitnessGen)
        ax.set_xlabel("generation num")
        ax.set_ylabel("fitness")
        ax.set_title("fitness results")

        ax = figVals.add_subplot(2, 1, 2)

        img = plotImg(ax, paramsAll[0], paramsAll[1], fitnessAll, params[0], params[1], "fitness value", binY=1000, binX=0.5)
        figVals.colorbar(img, shrink=0.4, aspect=5)
        plt.show()

