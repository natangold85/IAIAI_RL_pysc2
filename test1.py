from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import time

import random
import matplotlib.pyplot as plt

# Parameters
exploreProb = 0.9
learning_rate = 0.1
training_epochs = 500
training_sessions = 1000 
betaWeights = 0.3
betaBiases = 0.3

# Network Parameters
# n_hidden_1 = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
rangeOfValues = 3
n_input = 2
n_classes = n_input * 2
# n_classes = 1

n_hidden_1 = 256
n_hidden_2 = 256

qtableName = "test_simple_qtable"
resultsFName = "test_simple_results"

DQN_Name = "test_simple_dqn"
DQN_resultsFName = "test_simple_dqn_results"

# Store layers weight & bias
seed_now = time.time()
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed = seed_now)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # # Hidden fully connected layer with 256 neurons
    # layer_1 = tf.layers.dense(x, n_hidden_1, activation = tf.nn.relu)
    # layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation = tf.nn.relu)
    # out_layer = tf.layers.dense(layer_2, n_classes, activation = tf.nn.relu)

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.matmul(x, weights['w1']) + biases['b1']
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.matmul(layer_1, weights['w2']) + biases['b2']
    # Output fully connected layer with a neuron for each class  
    # make sigmoid in range of -1 to 1
    out_layer = tf.nn.sigmoid(tf.matmul(layer_2, weights['out'])) + biases['out']

    return out_layer

def RewardProp(rewards, discount = 0.5):
  if len(rewards) > 1:
    for i in range (len(rewards) - 2, -1, -1):
      if rewards[i] == 0:
          rewards[i] = rewards[i + 1] * discount

  
  return rewards
def ExtractExamples(num):
  s = np.zeros((num,n_input), dtype = int)
  labels = np.zeros((num,n_classes), dtype = float)
  
  return s, labels

def CalcModelValues(sIdx, a, stateModelValues, groupStateHist, groupActions, groupRewards):
  count = stateModelValues[sIdx][2][a]
  state2Cmp = stateModelValues[sIdx][0]
  valList = stateModelValues[sIdx][1][a]
  
  if count > 0:
    prevVal = valList[len(valList) - 1]
  else:
    prevVal = 0
  
  currSumVal = 0
  currCount = count
  for i in range(len(groupStateHist)):
    if np.array_equal(state2Cmp,groupStateHist[i]) and a == groupActions[i]:
      currCount += 1
      currSumVal += groupRewards[i]
  
  if currCount > count:
    nextVal = (prevVal * count + currSumVal) / currCount
    valList.append(nextVal)
    stateModelValues[sIdx][2][a] = currCount
  else:
    valList.append(prevVal)


  

class TEST:
  def __init__(self, sizeState, rangeOfValues = 1, valToEnd = 0, step2End = 20):
    self.numActions = sizeState * 2
    self.state = np.zeros(sizeState, dtype=int)
    self.env_name = "test"
    self.env_type = "1"

    self.minInitVal = valToEnd - rangeOfValues
    self.maxInitVal = valToEnd + rangeOfValues

    self.sizeState = sizeState
    self.valToEnd = valToEnd
    self.numStep = 0
    self.step2End = step2End

  def Terminal(self, state):
    win = True
    loss = False

    for i in range(0, self.sizeState):
      if state[i] != self.valToEnd:
         win = False
      if state[i] > self.maxInitVal or state[i] < self.minInitVal:
        loss = True

    if loss:
      return True, -1.0
    elif win:
      return True, 1.0

    if self.numStep == self.step2End:
      return True, 0
    else:
      return False, 0

  def PrintState(self):
    ret = ""
    for i in range(self.sizeState):
      ret += str(self.state[i]) + ','
    
    return ret

  def new_random_game(self):
    self.numStep = 0
    for i in range(0, self.sizeState):
      self.state[i] = random.randint(self.minInitVal, self.maxInitVal)

    terminal, r = self.Terminal(self.state)
    if (terminal):
      return self.new_random_game()

    return self.state.copy()

  def new_game(self, state):
    self.state = state.copy()
    return state

  def act(self, action):
    self.numStep += 1
    if action < self.sizeState:
      self.state[action] += 1
    else:
      action -= self.sizeState
      self.state[action] -= 1
    
    terminal, r = self.Terminal(self.state)

    return self.state.copy(), r, terminal

  def GetAllPossibleStates(self):
    allStates = []
    state = np.zeros(self.sizeState, dtype=int)
    self.FillStatesRec(allStates, state)
    return allStates

  def FillStatesRec(self, allStates, state, currIdx = 0):
    if currIdx == self.sizeState:
      terminalState, _ = self.Terminal(state)
      if not terminalState:
        allStates.append(state.copy())
    else:
      for i in range(self.minInitVal, self.maxInitVal + 1):
        state[currIdx] = i
        self.FillStatesRec(allStates, state, currIdx + 1)


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# The TD target value
y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
# Integer id of which action was selected
actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

batch_size = tf.shape(X)[0]

logits = multilayer_perceptron(X)


# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

gather_indices = tf.range(batch_size) * tf.shape(logits)[1] + actions_pl
action_predictions = tf.gather(tf.reshape(logits, [-1]), gather_indices)


losses = tf.squared_difference(y_pl, action_predictions)

weightsRegularizer = tf.nn.l2_loss(weights["w1"]) + tf.nn.l2_loss(weights["w2"])
biasesRegularizer = tf.nn.l2_loss(biases["b1"]) + tf.nn.l2_loss(biases["b2"])
# + betaWeights * weightsRegularizer + betaBiases * biasesRegularizer
loss_op = tf.reduce_mean(losses)

#loss_op = tf.reduce_mean(tf.square(Y - logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

rewardDict = {}

game = TEST(n_input, rangeOfValues)
gameRandom = TEST(n_input, rangeOfValues)

results = []
resultsRandom = []
w1 = sess.run(weights["w1"])
w2 = sess.run(weights["w2"])
w1All = np.zeros((training_sessions + 1, w1.shape[0], w1.shape[1]), dtype = float)
w2All = np.zeros((training_sessions + 1, w2.shape[0], w2.shape[1]), dtype = float)

w1All[0, :, :] = w1
w2All[0, :, :] =  w2

allStates = game.GetAllPossibleStates()
stateNNValues = []
stateModelValues = []
for s in allStates:
  av1 = []
  av2 = []
  count = []
  for a in range(game.numActions):
    av1.append([])
    av2.append([])
    count.append(0)
  
  stateNNValues.append([s, av1])
  stateModelValues.append([s.copy(), av2, count])

lossHistory = []
computedLoss = []
for group in range(training_sessions):
  sumReward = 0
  sumRewardRandom = 0
  groupStateHist = []
  groupRewards = []
  groupActions = []

  for epoch in range(training_epochs):
    hist = []
    actions = []
    rewards = []

    s = game.new_random_game()
    sRandom = gameRandom.new_game(s)
        
    terminalGame = False
    while not terminalGame:
      
      if np.random.uniform() > exploreProb:
        vals = logits.eval({X: s.reshape(1,n_input)}, session=sess)
        a = np.argmax(vals[0])      
      else:
        a = random.randint(0, game.numActions - 1)

      s_, r, terminalGame = game.act(a)

      hist.append(s)
      rewards.append(r)
      actions.append(a)
      s = s_

    sumReward += r

    terminalRandom = False
    while not terminalRandom:
      aRandom = random.randint(0, game.numActions - 1)
      s_, rRandom, terminalRandom = gameRandom.act(aRandom)

    sumRewardRandom += rRandom

    rewards = RewardProp(rewards)
    groupStateHist = groupStateHist + hist
    groupRewards = groupRewards + rewards
    groupActions = groupActions + actions


    for i in range(len(hist)):
      key = str(hist[i])
      if key not in rewardDict:
        rewardDict[key] = [[0], 0]
      else:
          rewardDict[key][0][len(rewardDict[key][0]) - 1] += rewards[i]
          rewardDict[key][1] += 1

  stateVec = np.array(groupStateHist).reshape(len(groupStateHist), n_input)
  rewardVec = np.array(groupRewards)
  actionVec = np.array(groupActions)

  for s, v in rewardDict.items():
    if rewardDict[s][1] > 0:
        rewardDict[s][0][len(rewardDict[s][0]) - 1] /= rewardDict[s][1]
        rewardDict[s][0].append(0)

  _, lossSingle = sess.run([train_op, loss_op], feed_dict={X: stateVec, y_pl: rewardVec, actions_pl: actionVec})

  lossHistory.append(lossSingle / len(groupStateHist))
  print(group)


  for s in range(len(stateNNValues)):
    key = stateNNValues[s][0]
    v = logits.eval({X: key.reshape(1,n_input)}, session=sess)
    for a in range(len(v[0])):
      stateNNValues[s][1][a].append(v[0][a])
      CalcModelValues(s, a, stateModelValues, groupStateHist, groupActions, groupRewards)
    


  # for sVal in stateModelValues:
  #   key = sVal[0]
  #   v = logits.eval({X: key.reshape(1,n_input)}, session=sess)
  #   for i in range(len(v[0])):
  #     sVal[1][i].append(v[0][i])

  w1All[group + 1, :, :] = sess.run(weights["w1"])
  w2All[group + 1, :, :] =  sess.run(weights["w2"])

  avgReward = sumReward / training_epochs
  avgRewardRandom = sumRewardRandom / training_epochs

  results.append(avgReward)
  resultsRandom.append(avgRewardRandom)

numCols = 2
numPlots = 2 + np.power(rangeOfValues * 2 + 1, n_input) - 1
numRows = int(numPlots / numCols)
def Advance2NextGraph(R, C, numCols):
  C += 1
  if C == numCols:
    R += 1
    C = 0
  return R, C

#f, ax = plt.subplots(numRows, numCols)
# idxR = 0
# idxC = 0
# for s in range(len(stateNNValues)):
  
#   leg = []
#   for i in range(len(stateNNValues[s][1])):
#     ax[idxR][idxC].plot(stateNNValues[s][1][i])
#     leg.append("nn-values_action " + str(i))
  
#   for i in range(len(stateNNValues[s][1])):
#     diff = np.power(np.array(stateModelValues[s][1][i]) - np.array(stateNNValues[s][1][i]), 2)
#     ax[idxR][idxC].plot(diff)
#     leg.append("mse_action " + str(i))

#   key = str(stateNNValues[s][0])
#   ax[idxR][idxC].legend(leg)
#   ax[idxR][idxC].set_title("for location " + key)
#   idxR, idxC = Advance2NextGraph(idxR, idxC, numCols)
#   #plt.ylim([-2,2])

f, ax = plt.subplots(2, 1)
ax[0].plot(lossHistory)
ax[0].set_title("loss history")


# plt.figure()
# legW = []
# for i in range(w1.shape[0]):
#   for o in range(w1.shape[1]):
#     plt.plot(w1All[:,i,o])
#     legW.append("w" + str(i) + '-' + str(o))
# plt.title("weight layer 1 change in time")
# plt.legend(legW)

# plt.figure()
# legW = []
# for i in range(w2.shape[0]):
#   for o in range(w2.shape[1]):
#     plt.plot(w2All[:,i,o])
#     legW.append("w" + str(i) + '-' + str(o))
# plt.title("weight layer 2 change in time")
# plt.legend(legW)

ax[1].plot(results)
ax[1].plot(resultsRandom)
ax[1].set_title("Results")
plt.show()
      