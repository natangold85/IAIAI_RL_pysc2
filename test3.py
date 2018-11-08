import numpy as np

import matplotlib.pyplot as plt
from utils_plot import PlotMeanWithInterval

from algo_geneticProgramming import GP_Params
from algo_geneticProgramming import RationalParamType
from algo_geneticProgramming import Population
from algo_geneticProgramming import ParamsState

class GPGame:
    def __init__(self, paramsState, params): 
        self.paramsState = paramsState   
        self.populationMngr = Population(paramsState, params)
        
        self.idealVals = [0.5, 10, -1.0]

        self.avgFitnessAll = []
        self.avgStateAll = []
        self.bestFitnessAll = []
        self.bestStateAll = []

        self.avgFitness = []
        self.avgState = []
        self.bestFitness = []
        self.bestState = []


    def EndEpoch(self):
        self.avgFitnessAll.append(self.avgFitness)
        self.avgStateAll.append(self.avgState)
        self.bestFitnessAll.append(self.bestFitness)
        self.bestStateAll.append(self.bestState)

        self.avgFitness = []
        self.avgState = []
        self.bestFitness = []
        self.bestState = []

        self.populationMngr.InitPopulation()
    
    def GetPopulation(self):
        return self.populationMngr.GetPopulation()

    def Generation(self):
        population = self.populationMngr.GetPopulation()
        fitnessDict = {}
        for s in population:
            fitness = Fitness(s)
            fitnessDict[str(s)] = fitness

        maxFitness, stateMax = self.populationMngr.SetFitness(fitnessDict)

        self.avgFitness.append(np.average(self.populationMngr.fitness))
        self.avgState.append(np.average(self.populationMngr.population, axis=0))

        self.bestFitness.append(maxFitness)
        self.bestState.append(stateMax)

        self.populationMngr.Cycle()

        return self.avgFitness[-1] < 0.99999

    def Plot(self, title, show=False):
        
        size = len(self.bestFitnessAll[0])
        figVals = plt.figure(figsize=(19.0, 11.0))
        plt.suptitle(title)
        
        bestGenState = np.array(self.bestStateAll)
        avgGenState = np.array(self.avgStateAll)

        numGen = np.arange(size)

        ax = figVals.add_subplot(3, 2, 1)  
        ax.set_title("fitness value") 
        
        PlotMeanWithInterval(numGen, np.average(self.bestFitnessAll, axis=0), np.std(self.bestFitnessAll, axis=0))
        PlotMeanWithInterval(numGen, np.average(self.avgFitnessAll, axis=0), np.std(self.avgFitnessAll, axis=0))

        ax.legend(["best fitness", "avg fitness"])

        for i in range(self.paramsState.Size()):
            ax = figVals.add_subplot(3, 2, i+2)   
            
            idealVal = np.ones(size) * self.idealVals[i]
            bestGenVal = bestGenState[:, :, i]
            avgGenVal = avgGenState[:, :, i]

            PlotMeanWithInterval(numGen, np.average(bestGenVal, axis=0), np.std(bestGenVal, axis=0))
            PlotMeanWithInterval(numGen, np.average(avgGenVal, axis=0), np.std(avgGenVal, axis=0))
            ax.plot(numGen, idealVal)
 
            ax.set_title(self.paramsState.ParamName(i) + " value") 
            ax.legend(["best state val", "avg population val", "ideal val"])

        if show:
            plt.show()

def Fitness(s):
    fitness = s[0] * s[1] + (s[2] ** 2) / s[1] - (s[2] * s[0] + s[1] / s[2] ) - s[3]

    return fitness

def CreateParamsState(params):
    p1 = RationalParamType(name="GPGame_p1", minVal=0, maxVal=1.0, breedProbType=params.breedProbType)
    p2 = RationalParamType(name="GPGame_p2", minVal=0, maxVal=50, breedProbType=params.breedProbType)
    p3 = RationalParamType(name="GPGame_p3", minVal=-10, maxVal=10, breedProbType=params.breedProbType)
    p4 = RationalParamType(name="GPGame_p3", minVal=-10, maxVal=10, breedProbType=params.breedProbType)

    paramsList = [p1,p2,p3,p4]
    return ParamsState(paramsList)

if __name__ == "__main__":


    basicParams = GP_Params(populationSize=100, breedPriority=False, breedProbType="uniform")
    envBasic = GPGame(CreateParamsState(basicParams), basicParams)
    
    normalParams = GP_Params(populationSize=100, breedPriority=False, breedProbType="normal")
    envNormal = GPGame(CreateParamsState(normalParams), normalParams)
    
    priorityParams = GP_Params(populationSize=100, breedPriority=True, breedProbType="uniform")
    envPriority = GPGame(CreateParamsState(priorityParams), priorityParams)
    
    priority_normalParams = GP_Params(populationSize=100, breedPriority=True, breedProbType="normal")
    envPriority_Normal = GPGame(CreateParamsState(priority_normalParams), priority_normalParams)
    
    for epoch in range(20):
        for i in range(100):
            envBasic.Generation()
            envNormal.Generation()
            envPriority.Generation()
            envPriority_Normal.Generation()

        envBasic.EndEpoch()
        envNormal.EndEpoch()
        envPriority.EndEpoch()
        envPriority_Normal.EndEpoch()
        

    envBasic.Plot("simple GP")
    envNormal.Plot("breeding with normal distribution")
    envPriority.Plot("priority to breed depend on fitness")
    envPriority_Normal.Plot("priority to breed depend on fitness & breeding with normal distribution", show=True)
