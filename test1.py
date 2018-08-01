from utils_decisionMaker import LearnWithReplayMngr
from utils_dqn import DQN_WITH_TARGET_PARAMS

from utils_results import PlotMngr

from maze_game import MazeGame

import sys

holes = []
holes.append([1,1])
holes.append([5,3])
holes.append([4,8])
holes.append([7,9])
holes.append([8,2])
holes.append([6,1])
holes.append([3,5])
holes.append([2,3])
holes.append([8,6])
holes.append([0,2])
env = MazeGame(10, holes)

numTrials2CmpResults = 500
p = DQN_WITH_TARGET_PARAMS(stateSize=env.stateSize, numActions=env.numActions, numTrials2CmpResults=numTrials2CmpResults)
dqn = LearnWithReplayMngr("DQN_WithTarget", p, decisionMakerName = 'test_target', resultFileName = 'results', historyFileName = 'hist', directory = "test1")

runTypeArg = ["test1"]
resultFnames = ['results']
directoryNames = ["test1"]

plotName = "results.png"

for i in range(100000):
  s = env.newGame()
  terminal = False
  sumR = 0
  numSteps = 0
  
  while not terminal:
    a = dqn.choose_action(s)
    s_, r, terminal = env.step(s,a)
    dqn.learn(s, a, r, s_, terminal)
    sumR += r
    numSteps += 1
    s = s_

  isSaved = dqn.end_run(sumR,sumR,numSteps)  
  print("num runs =", dqn.NumRuns(), "num target runs =", dqn.decisionMaker.NumRunsTarget())  
  if dqn.NumRuns() > numTrials2CmpResults and isSaved:
    plot = PlotMngr(resultFnames, directoryNames, runTypeArg, directoryNames[0])
    plot.Plot(numTrials2CmpResults)



