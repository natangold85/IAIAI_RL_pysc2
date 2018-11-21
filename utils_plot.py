import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# def PlotMeanWithInterval(mean, interval, color_mean=None):
#     # plot the shaded range of the confidence intervals
#     plt.fill_between(range(mean.shape[0]), mean + interval, mean - interval, alpha=.5)
#     # plot the mean on top
#     plt.plot(mean)

def PlotMeanWithInterval(x, y, interval, color=None):
    if color != None:
        # plot the shaded range of the confidence intervals
        plt.fill_between(x, y + interval, y - interval, color=color, alpha=.5)
        # plot the mean on top
        plt.plot(x, y, color=color)
    else:
        # plot the shaded range of the confidence intervals
        plt.fill_between(x, y + interval, y - interval, alpha=.5)
        # plot the mean on top
        plt.plot(x, y)

def create_nnGraphs(superAgent, agent2Check, statesIdx, actions2Check, plotTarget=False, numTrials = -1, saveGraphs = False, showGraphs = False, dir2Save = "./", maxSize2Plot=20000):
    plotType = "target" if plotTarget else "current"

    figVals = None
    isFigVals = False
    figDiff = None
    isFigDiff = False

    idxX = statesIdx[0]
    idxY = statesIdx[1]
    
    agent = superAgent.GetAgentByName(agent2Check)
    dm = agent.GetDecisionMaker()

    if plotTarget:
        numRuns = dm.decisionMaker.NumRunsTarget()
    else:
        numRuns = numTrials if numTrials >= 0 else dm.decisionMaker.NumRuns()

    xName = agent.StateIdx2Str(idxX)
    yName = agent.StateIdx2Str(idxY)

    actionsPoints = {}

    # extracting nn vals for current nn and target nn

    for a in actions2Check:
        actionsPoints[a] = [[], [], [], []]

    sizeHist = len(dm.historyMngr.transitions["a"])
    size2Plot = min(sizeHist, maxSize2Plot)
    for i in range(size2Plot):
        s = dm.DrawStateFromHist(realState=False)
        validActions = agent.ValidActions()
        vals = dm.ActionsValues(s, validActions, targetValues=plotTarget)

        agent.current_scaled_state = s
        
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
 
    if saveGraphs:
        if isFigVals:
            figVals.savefig(dir2Save + plotType + "DQN_" + str(numRuns))
        if isFigDiff:
            figDiff.savefig(dir2Save + plotType + "DQNDiff_" + str(numRuns))

    if showGraphs:
        plt.show()



def plotImg(ax, x, y, z, xName, yName, title, minZ = None, maxZ = None, binX=1, binY=1):
    if len(x) == 0:
        return None
    imSizeX = math.ceil((np.max(x) - np.min(x)) / binX) + 1
    imSizeY = math.ceil((np.max(y) - np.min(y)) / binY) + 1

    offsetX = int(np.min(x) / binX) * binX
    offsetY = int(np.min(y) / binY) * binY

    mat = np.zeros((imSizeY, imSizeX), dtype=float)
    mat.fill(np.nan)

    duplicateDict = {}
    for i in range(len(x)):
        yIdx = int(round((y[i] - offsetY) / binY))
        xIdx = int(round((x[i] - offsetX) / binX))
        if np.isnan(mat[yIdx, xIdx]):
            mat[yIdx, xIdx] = z[i]
        else:
            key = xIdx + yIdx * imSizeX
            if key not in duplicateDict.keys():
                duplicateDict[key] = [mat[yIdx, xIdx]]
            
            duplicateDict[key].append(z[i])
            mat[yIdx, xIdx] = np.average(duplicateDict[key])



    if minZ == None:
        minZ = np.nanmin(mat)
    if maxZ == None:
        maxZ = np.nanmax(mat)

    img = ax.imshow(mat, cmap=plt.cm.coolwarm, vmin=minZ, vmax=maxZ)

    if (np.max(x) - np.min(x)) / binX < 10:
        xTick = np.arange(0, int((np.max(x) - offsetX) / binX) + 1)
    else:
        xTick = np.arange(0, int((np.max(x) - offsetX) / binX) + 1, 4) 

    if (np.max(y) - np.min(y)) / binY < 10:
        yTick = np.arange(int((np.max(y) - offsetY) / binY) + 1)
    else:
        yTick = np.arange(int((np.max(y) - offsetY) / binY) + 1, 4)

    ax.set_xticks(xTick)
    ax.set_xticklabels(xTick * binX + offsetX)
    
    ax.set_yticks(yTick)
    ax.set_yticklabels(yTick * binY + offsetY)
    
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