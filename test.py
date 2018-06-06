from multiprocessing import Process, Manager

def f(d):
    d[1] += '1'
    d['2'] += 2

if __name__ == '__main__':
    manager = Manager()

    d = manager.dict()
    d[1] = '1'
    d['2'] = 2

    p1 = Process(target=f, args=(d,))
    p2 = Process(target=f, args=(d,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print (d)




# import random
# import tensorflow as tf
# import numpy as np

# from utils_tables import TableMngr
# from utils_dqn import DQN

# qtableName = "test_simple_qtable"
# resultsFName = "test_simple_results"

# DQN_Name = "test_simple_dqn"
# DQN_resultsFName = "test_simple_dqn_results"

# class TEST:
#   def __init__(self, minInitVal, maxInitVal, sizeState, valToEnd):
#     self.action_size = sizeState * 2
#     self.state = np.zeros(sizeState, dtype=int)
#     self.env_name = "test"
#     self.env_type = "1"

#     self.minInitVal = minInitVal
#     self.maxInitVal = maxInitVal

#     self.sizeState = sizeState
#     self.valToEnd = valToEnd

#   def Terminal(self):
#     win = True
#     loss = False

#     for i in range(0, self.sizeState):
#       if self.state[i] != self.valToEnd:
#          win = False
#       if self.state[i] > self.maxInitVal or self.state[i] < self.minInitVal:
#         loss = True

#     if loss:
#       print("loss, state =", self.PrintState())
#       return True, -1.0
#     elif win:
#       print("win, state =", self.PrintState())
#       return True, 1.0

#     return False, 0

#   def PrintState(self):
#     ret = ""
#     for i in range(self.sizeState):
#       ret += str(self.state[i]) + ','
    
#     return ret

#   def new_random_game(self):
#     for i in range(0, self.sizeState):
#       self.state[i] = random.randint(self.minInitVal, self.maxInitVal)
    
#     terminal, r = self.Terminal()
#     return self.state.copy(), r, 0, terminal

#   def act(self, action):
#     # self.PrintState()
#     # print(" --->", end = ' ')
#     if action < self.sizeState:
#       self.state[action] += 1
#     else:
#       action -= self.sizeState
#       self.state[action] -= 1
    
#     terminal, r = self.Terminal()

#     # self.PrintState()
#     # print("")
#     return self.state.copy(), r, terminal

# game = TEST(0,10,2,5)
# tables = TableMngr(game.action_size, qtableName, resultsFName)

# while True:
#     s, r, a, terminal = game.new_random_game()
#     while not terminal:
#         a = tables.choose_action(str(s))
#         s_, r, terminal = game.act(a)
#         tables.learn(str(s), a, r, str(s_))
#         s = s_
    
#     tables.end_run(r)
        

