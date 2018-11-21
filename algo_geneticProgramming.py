import numpy as np

class GP_Params:
    def __init__(self, populationSize, topPct=0.1, bottomPct=0.1, breedPriority=False, breedProbType="uniform", pctMutation=0.02, fitnessInitVal=np.nan):
        self.populationSize = populationSize
        self.topPct = topPct
        self.bottomPct = bottomPct
        self.breedPriority = breedPriority
        self.breedProbType = breedProbType
        self.pctMutation = pctMutation
        self.fitnessInitVal = fitnessInitVal


class ParamsType:
    def __init__(self, name, paramType):
        self.name = name
        self.paramType = paramType


class RationalParamType(ParamsType):
    def __init__(self, name, minVal, maxVal, floatingValueValid=True, breedProbType="uniform", initOptions=None):
        super(RationalParamType, self).__init__(name, "ratio")
        self.minVal=minVal
        self.maxVal=maxVal
        self.floatingValueValid = floatingValueValid
        self.breedProbType = breedProbType

        self.initOptions = initOptions
        if self.breedProbType == "normal":
            self.sigma = 0.3

    def Breed(self, p1Val, p2Val, pctMutation):
        if np.random.uniform() > pctMutation:
            maxVal = max(p1Val, p2Val)
            minVal = min(p1Val, p2Val)
            if self.breedProbType == "uniform":
                newVal = np.random.uniform() * (maxVal-minVal) + minVal
            elif self.breedProbType == "normal":
                # slice val space in the middle and reverse normal prob to highlight edges
                diff2Half = (maxVal - minVal) / 2
                v = np.random.normal(0, self.sigma, 1).squeeze()
                # deciding on which prob 2 go
                fromVal = minVal if v > 0 else maxVal
                newVal = fromVal + v * diff2Half
        else:
            newVal = self.New()
        
        return newVal if self.floatingValueValid else int(newVal)

    def New(self):
        if self.initOptions == None:
            val = np.random.uniform() * (self.maxVal-self.minVal) + self.minVal
            return val if self.floatingValueValid else int(val)
        else:
            return np.random.choice(self.initOptions)


class TraitParamType(ParamsType):
    def __init__(self, name, allVals):
        super(TraitParamType, self).__init__(name, "trait")
        self.allVals=allVals

    def Breed(self, p1Val, p2Val, pctMutation):
        if np.random.uniform() > pctMutation:
            return p1Val if np.random.randint(2) == 0 else p2Val
        else:
            return np.random.choice(self.allVals)
        
    def New(self):
        return np.random.choice(self.allVals)

class ParamsState:
    def __init__(self, paramsList = []):
        self.paramsList = paramsList
    
    def AddParam(self, param):
        paramExist = False
        for i in range(len(self.paramsList)):
            paramExist |= self.paramsList[i].name == param.name

        if not paramExist:
            self.paramsList.append(param)

    def InitRandomState(self):
        s = []
        for param in self.paramsList:
            s.append(param.New())
        return s

    def Breed(self, s1, s2, pctMutation):
        son = []
        for i in range(len(self.paramsList)):
            son.append(self.paramsList[i].Breed(s1[i], s2[i], pctMutation))    
        
        return son
    
    def ParamName(self, idx):
        return self.paramsList[idx].name
    
    def Size(self):
        return len(self.paramsList)
    
    def Params2Dict(self, params):
        paramsDict = {}
        for i in range(len(params)):
            paramsDict[self.paramsList[i].name] = params[i]

        return paramsDict

class Population:
    def __init__(self, paramsState, gpParams):
        self.paramsState = paramsState
        self.size = gpParams.populationSize

        self.breedPriority = gpParams.breedPriority


        numSurvival = int(self.size * gpParams.topPct)
        numDrops = int(self.size * gpParams.bottomPct)
        numBreeds = int(self.size - numSurvival - numDrops)

        self.survivalIdx = np.arange(numSurvival)
        self.breedIdx = np.arange(numSurvival, numSurvival + numBreeds)
        self.dropIdx = np.arange(numSurvival + numBreeds, self.size)

        self.pctMutation = gpParams.pctMutation

        self.fitnessInitVal = gpParams.fitnessInitVal

        self.InitPopulation()
    
    def Size(self):
        return len(self.population)

    def Population(self, population, fitness=[]):
        if len(population) != self.size:
            print("population size is wrong in load")
            return
        
        self.population = population
        if len(fitness) > 0:
            self.fitness = fitness
        else:
            self.fitness = np.ones(self.size, float) * self.fitnessInitVal
    
    def InitPopulation(self):
        self.population = []
        self.fitness = np.ones(self.size, float) * self.fitnessInitVal

        for i in range(self.size):
            self.population.append(self.paramsState.InitRandomState())
    
    def GetPopulation(self):
        return self.population.copy()

    def SetFitness(self, fitnessDict):
        for i in range(len(self.population)):
            key4Dict = str(self.population[i])
            if key4Dict in fitnessDict:
                self.fitness[i] = fitnessDict[key4Dict]
        
        idxMax = np.argmax(self.fitness)
        return self.fitness[idxMax], self.population[idxMax]

    def Breed(self, pop2Breed, fitness):
        if self.breedPriority:
            priority = fitness - min(fitness) + 0.1
            logPriority = np.log(priority * 10 + 1.0)
            probLogPriority = logPriority / sum(logPriority)

            idxArray = np.arange(len(pop2Breed))
            idx1 = np.random.choice(idxArray, p=probLogPriority)
            probLogPriority[idx1] = 0
            probLogPriority /= sum(probLogPriority)
            idx2 = np.random.choice(idxArray, p=probLogPriority)       

        else:
            idx1 = np.random.randint(0, len(pop2Breed))
            idx2 = np.random.randint(0, len(pop2Breed) - 1)
            idx2 = idx2 if idx1 != idx2 else len(pop2Breed) - 1

        return self.paramsState.Breed(pop2Breed[idx1], pop2Breed[idx2], self.pctMutation)

    def Cycle(self):
        idxSorted = np.argsort(-1 * self.fitness)
        self.fitness = self.fitness[idxSorted]
        self.population = [self.population[i] for i in idxSorted]
     
        newPopulation = [self.population[i] for i in self.survivalIdx]

        pop2Breed = newPopulation.copy()
        pop2Breed += [self.population[i] for i in self.breedIdx]

        fitnessOfBreed = self.fitness[0:len(pop2Breed)]

        for i in range(len(newPopulation), self.size):
            newPopulation.append(self.Breed(pop2Breed, fitnessOfBreed))
            self.fitness[i] = self.fitnessInitVal

        self.population = newPopulation