import os
import numpy as np

from algo_geneticProgramming import GP_Params
from algo_geneticProgramming import RationalParamType
from algo_geneticProgramming import Population
from algo_geneticProgramming import ParamsState

class Calibration:
    def __init__(self, paramsState, params, path2Data, directoryName, numGameThreads):
        self.populationMngr = Population(paramsState, params)
        self.path2Data = path2Data
        self.paramsFName = "paramsState.txt"
        self.numGameThreads = numGameThreads

    def ReadFittness(self, population):
        fitnessDict  = {}
        dirName = self.path2Data
        if os.path.isdir(dirName) and os.path.isfile(dirName + "/" + self.paramsFName):
            currDirectoryDict = eval(open(dirName + "/" + self.paramsFName, "r+").read())
            fitnessDict.update(currDirectoryDict) 

        idx = 0
        currDirName = dirName + "_" + str(idx)
        while os.path.isdir(currDirName):   
            if os.path.isfile(currDirName + "/" + self.paramsFName):
                currDirectoryDict = eval(open(currDirName + "/" + self.paramsFName, "r+").read())
                fitnessDict.update(currDirectoryDict) 
            
            idx += 1
            currDirName = dirName + "_" + str(idx)
        
        return fitnessDict

    def Generation(self):
        population = self.populationMngr
        # read fitness for current population
        fitnessDict = self.ReadFittness(population)
        
        self.populationMngr.SetFitness(fitnessDict)
        
        # go to next generation
        self.populationMngr.Cycle()

        # train population and save population in file
        self.TrainPopulation()

    
    def TrainPopulation(self):
        pass


def GenProgCalibration(dmTypes, trainAgent):

    params = GP_Params(populationSize=100, fitnessInitVal=None)
    paramsState = ParamsState([])
    calib = Calibration(paramsState, params)
    
    agent = SuperAgent(decisionMaker=decisionMaker, isMultiThreaded=isMultiThreaded, dmTypes=dmTypes, playList=playList, trainList=trainList, useMapRewards=useMapRewards, dmCopy=dmCopy)
