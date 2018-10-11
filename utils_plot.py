import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def create_nnGraphs(superAgent, agent2Check, statesIdx, actions2Check, plotTarget=False, numTrials = -1, saveGraphs = False, showGraphs = False, dir2Save = "./", maxSize2Plot=20000):
    plotType = "target" if plotTarget else "current"

    figVals = None
    figDiff = None

    idxX = statesIdx[0]
    idxY = statesIdx[1]
    
    agent = superAgent.GetAgentByName(agent2Check)
    dm = agent.GetDecisionMaker()

    numRuns = numTrials if numTrials >= 0 else dm.decisionMaker.NumRuns()
    numRunsTarget = dm.decisionMaker.NumRunsTarget()

    xName = agent.StateIdx2Str(idxX)
    yName = agent.StateIdx2Str(idxY)

    actionsPoints = {}

    # extracting nn vals for current nn and target nn

    for a in actions2Check:
        actionsPoints[a] = [[], [], [], []]

    sizeHist = len(dm.historyMngr.transitions["a"])
    size2Plot = min(sizeHist, maxSize2Plot)
    for i in range(size2Plot):
        s = dm.DrawStateFromHist()
        vals = dm.ActionValuesVec(s, targetValues=plotTarget)

        agent.current_scaled_state = s
        validActions = agent.ValidActions()
        
        if xName == "min" or xName == "MIN":
            s[idxX] = int(s[idxX] / 25) 
        if yName == "min" or yName == "MIN":
            s[idxY] = int(s[idxY] / 25) 


        for a in actions2Check:
            if a in validActions:
                add_point(s[idxX], s[idxY], vals[a], actionsPoints[a])
            else:
                add_point(s[idxX], s[idxY], np.nan, actionsPoints[a])

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

        for a2Idx in range(a1Idx + 1, len(actions2Check)):
            a2 = actions2Check[a2Idx]
            z1 = np.array(actionsPoints[a1][3])
            z2 = np.array(actionsPoints[a2][3])
            zDiff = z1 - z2
            maxZ = np.max(np.abs(zDiff))
            ax = figDiff.add_subplot(numRows, 2, idxPlot)
            title = "values for differrence = " + agent.Action2Str(a1, onlyAgent=True) + " - " + agent.Action2Str(a2, onlyAgent=True)
            img = plotImg(ax, x, y, zDiff, xName, yName, title, minZ=-maxZ, maxZ=maxZ)
            figDiff.colorbar(img, shrink=0.4, aspect=5)
            idxPlot += 1
 
    if saveGraphs:
        if figVals != None:
            figVals.savefig(dir2Save + plotType + "DQN_" + str(numRuns))
        if figDiff != None:
            figDiff.savefig(dir2Save + plotType + "DQNDiff_" + str(numRuns))

    if showGraphs:
        plt.show()

def plotImg(ax, x, y, z, xName, yName, title, minZ = None, maxZ = None):
    imSizeX = np.max(x) - np.min(x) + 1
    imSizeY = np.max(y) - np.min(y) + 1

    offsetX = np.min(x)
    offsetY = np.min(y)

    mat = np.zeros((imSizeY, imSizeX), dtype=float)
    mat.fill(np.nan)

    for i in range(len(x)):
        # print("[", x[i], y[i], "] =", z[i])
        mat[y[i] - offsetY, x[i] - offsetX] = z[i]

    if minZ == None:
        minZ = np.min(mat)
    if maxZ == None:
        maxZ = np.max(mat)

    img = ax.imshow(mat, cmap=plt.cm.coolwarm, vmin=minZ, vmax=maxZ)

    if np.max(x) - np.min(x) < 10:
        xTick = np.arange(np.min(x), np.max(x) + 1)
    else:
        xTick = np.arange(np.min(x), np.max(x) + 1, 4)

    if np.max(y) - np.min(y) < 10:
        yTick = np.arange(np.min(y), np.max(y) + 1)
    else:
        yTick = np.arange(np.min(y), np.max(y) + 1, 4)

    ax.set_xticks(xTick)
    ax.set_yticks(yTick)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_title(title)

    return img

def add_point(x, y, val, actionVec):
    for i in range(len(actionVec[0])):
        if x == actionVec[0][i] and y == actionVec[1][i]:
            actionVec[2][i].append(val)
            return
    
    actionVec[0].append(x)
    actionVec[1].append(y)
    actionVec[2].append([val])
    actionVec[3].append(0)