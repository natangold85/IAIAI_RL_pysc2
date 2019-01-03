import numpy as np
from utils_history import GetHistoryFromFile

transitions = GetHistoryFromFile("history")

terminals = np.array(transitions["terminal"])
rewards = np.array(transitions["r"])

idxTerm = terminals.nonzero()[0]
idxWon = (rewards > 0).nonzero()[0]
idxLoss = (rewards < 0).nonzero()[0]

idx = np.random.choice(idxWon)
idxAllTerminal = (idxTerm == idx).nonzero()[0][0]
idxStart = idxTerm[idxAllTerminal - 1] + 1
idxEnd = idx + 1
first = True
s_ = None
count = 0
np.set_printoptions(precision=2, suppress=True)
for i in range(idxStart, idxEnd):
    #print("s = ", transitions["s"][i], "s_ = ", transitions["s_"][i])
    s = transitions["s"][i]
    if not first:
        if not np.array_equal(s, s_):
            print(i, ":", s_, " || ", s)
        else:
            count += 1
    s_ = transitions["s_"][i]
    a = transitions["a"][i]
    # base action
    if a <= 1:
       print(s[:7], "action =", transitions["a"][i])
   
    # scout ation
    # if a > 1 and a < 6:
    #     print(s[4: ], "action =", transitions["a"][i])
    first = False