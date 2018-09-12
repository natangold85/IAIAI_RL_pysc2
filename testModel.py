import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

histFile = "replayHistory_buildOnly.gz"

numSDLimit = 0
numPositiveRec = 0

histTable = pd.read_pickle(histFile, compression='gzip')

rewards = []

actionBuildBarracks = 3
actionBuildReactor = 5
numActions = 7

idxMinerals = 1
idxBarracks = 5
idxBarracksInBuilt = 11
idxReactor = 7
idxReactorInBuilt = 13

allActions = [0] * numActions
countBuildBarrackSuccessfulActions = 0
mineralsBarracksSuccessful = [0.0, 0.0]
barracksBuildingsSuccessful = [0.0, 0.0]
barracksInProgressSuccessful = [0.0, 0.0]

countBuildBarrackFailedActions = 0
mineralsBarracksFailed = [0.0, 0.0]
barracksBuildingsFailed = [0.0, 0.0]
barracksInProgressFailed = [0.0, 0.0]

for i in range(len(histTable["r"])):
    allActions[histTable["a"][i]] += 1
    if histTable["a"][i] == actionBuildBarracks:
        if histTable["s_"][i][idxMinerals] < histTable["s"][i][idxMinerals]:
            countBuildBarrackSuccessfulActions += 1
            
            mineralsBarracksSuccessful[0] += histTable["s"][i][idxMinerals]
            mineralsBarracksSuccessful[1] += histTable["s_"][i][idxMinerals]

            barracksBuildingsSuccessful[0] += histTable["s"][i][idxBarracks]
            barracksBuildingsSuccessful[1] += histTable["s_"][i][idxBarracks]

            barracksInProgressSuccessful[0] += histTable["s"][i][idxBarracksInBuilt]
            barracksInProgressSuccessful[1] += histTable["s_"][i][idxBarracksInBuilt]
        else:
            countBuildBarrackFailedActions += 1
            
            mineralsBarracksFailed[0] += histTable["s"][i][idxMinerals]
            mineralsBarracksFailed[1] += histTable["s_"][i][idxMinerals]

            barracksBuildingsFailed[0] += histTable["s"][i][idxBarracks]
            barracksBuildingsFailed[1] += histTable["s_"][i][idxBarracks]

            barracksInProgressFailed[0] += histTable["s"][i][idxBarracksInBuilt]
            barracksInProgressFailed[1] += histTable["s_"][i][idxBarracksInBuilt]
        
    if histTable["r"][i] > 1 and not histTable["terminal"][i]:
        print(histTable["r"][i])

for i in range(2):
    mineralsBarracksSuccessful[i] /= countBuildBarrackSuccessfulActions
    barracksBuildingsSuccessful[i] /= countBuildBarrackSuccessfulActions
    barracksInProgressSuccessful[i] /= countBuildBarrackSuccessfulActions

    mineralsBarracksFailed[i] /= countBuildBarrackFailedActions
    barracksBuildingsFailed[i] /= countBuildBarrackFailedActions
    barracksInProgressFailed[i] /= countBuildBarrackFailedActions

print("\n\nBuild Barracks Action:")
print("\nSUCCESSFUL build barracks actions =")
print("num barracks build actions =", countBuildBarrackSuccessfulActions)
print("minerals s =", mineralsBarracksSuccessful[0], "s_ =", mineralsBarracksSuccessful[1])
print("barracks buildings s =", barracksBuildingsSuccessful[0], "s_ =", barracksBuildingsSuccessful[1])
print("barracks buildings in progress s =", barracksInProgressSuccessful[0], "s_ =", barracksInProgressSuccessful[1])

print("\nFAILED build barracks actions =")
print("num barracks build actions =", countBuildBarrackFailedActions)
print("minerals s =", mineralsBarracksFailed[0], "s_ =", mineralsBarracksFailed[1])
print("barracks buildings s =", barracksBuildingsFailed[0], "s_ =", barracksBuildingsFailed[1])
print("barracks buildings in progress s =", barracksInProgressFailed[0], "s_ =", barracksInProgressFailed[1])

print("\n\nBuild Reactor Action:")
print("\nSUCCESSFUL build reactor actions =")
print("num barracks build actions =", countBuildBarrackSuccessfulActions)
print("minerals s =", mineralsBarracksSuccessful[0], "s_ =", mineralsBarracksSuccessful[1])
print("barracks buildings s =", barracksBuildingsSuccessful[0], "s_ =", barracksBuildingsSuccessful[1])
print("barracks buildings in progress s =", barracksInProgressSuccessful[0], "s_ =", barracksInProgressSuccessful[1])

print("\nFAILED build reactor actions =")
print("num barracks build actions =", countBuildBarrackFailedActions)
print("minerals s =", mineralsBarracksFailed[0], "s_ =", mineralsBarracksFailed[1])
print("barracks buildings s =", barracksBuildingsFailed[0], "s_ =", barracksBuildingsFailed[1])
print("barracks buildings in progress s =", barracksInProgressFailed[0], "s_ =", barracksInProgressFailed[1])
