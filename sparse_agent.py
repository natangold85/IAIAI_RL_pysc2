import random
import math
import os.path
import logging
import traceback

#udp
import socket
import threading

import numpy as np
import pandas as pd
import time
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

#player general info
_MINERALS = 1
_VESPENE = 2


_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = np.zeros(23, dtype=np.int32, order='C')
        self.previous_supply_depot_during_build = 0
        self.previous_barracks_during_build = 0
        self.cc_y = None
        self.cc_x = None
        self.sentToDecisionMakerAsync = None
        self.returned_action_from_decision_maker = -1
        self.move_number = 0
        self.current_state_for_decision_making = None
        self.step_num = 0
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
            #print(self.qlearn.q_table.head())
            #exit()
        
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]
    
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)
    def sendToDecisionMaker(self):
        byte_array_current_state = self.current_state_for_decision_making.tobytes()
        result = -1
        data = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('172.23.40.129',5432)  #('localhost', 10000)
        message = byte_array_current_state
        try:
            # Send data
            print('sending ....')
            sent = sock.sendto(message, server_address)
            print('sent {} bytes'.format(sent))
            # Receive response
            sock.setblocking(0)
            sock.settimeout(10)
            data, server = sock.recvfrom(5432)
            print('recieved {} bytes'.format(len(data)))
            result = 0
            if(data != None):
                result = int(data[0])
                #for b in data:
                #    result = result * 256 + int(b)
        except (ConnectionResetError, socket.timeout):
            pass
        finally:
            sock.close()
          #  if(type(data) != 'bytes' or len(data) == 0):
          #      raise RuntimeError('error: wrong  data - {}  of type {}'.format(data, type(data)))
        self.returned_action_from_decision_maker = result
        print('self.returned_action_from_decision_maker =  {}'.format(self.returned_action_from_decision_maker))
        return result

    def step(self, obs):
        super(SparseAgent, self).step(obs)
        self.step_num += 1
        print(obs.observation.to_string())
        #print('**** step -  {}  current state - {}'.format(self.step_num, self.current_state_for_decision_making))
        #print('**** step -  {}  observation - {}'.format(self.step_num, obs))
        #time.sleep(0.25)
        try:
            if obs.last():
                reward = obs.reward
                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
                self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip') 
                self.previous_state = np.zeros(23, dtype=np.int32, order='C')
                self.previous_action = None
                self.previous_supply_depot_during_build = 0
                self.previous_barracks_during_build = 0
                self.move_number = 0
                self.sentToDecisionMakerAsync = None
                self.returned_action_from_decision_maker = -1
                self.current_state_for_decision_making = None
                self.cc_y = None
                self.cc_x = None
                self.step_num = 0
                return actions.FunctionCall(_NO_OP, [])
            
            unit_type = obs.observation['screen'][_UNIT_TYPE]

            if obs.first():
                player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
                self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
                self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                self.previous_action = None
                self.previous_state = np.zeros(23, dtype=np.int32, order='C')
                self.previous_supply_depot_during_build = 0
                self.previous_barracks_during_build = 0
                self.sentToDecisionMakerAsync = None
                self.returned_action_from_decision_maker = -1
                self.move_number = 0
                self.current_state_for_decision_making = None

            cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
            cc_count = 1 if cc_y.any() else 0
        
            depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
            supply_depot_count = int(round(len(depot_y) / 69))

            barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
            barracks_count = int(round(len(barracks_y) / 137))
                
            if self.move_number == 0:
                self.move_number += 1
                
                if(self.previous_state[1] < supply_depot_count):
                    self.previous_supply_depot_during_build -=  supply_depot_count - self.previous_state[1]
                if(self.previous_state[1] < barracks_count):
                    self.previous_barracks_during_build -=  supply_depot_count - self.previous_state[1]

                current_state_counter = 0
                current_state = np.zeros(21, dtype=np.int32, order='C')
                current_state[current_state_counter] = cc_count
                current_state_counter += 1
                current_state[current_state_counter] = supply_depot_count
                current_state_counter += 1
                current_state[current_state_counter] = barracks_count
                current_state_counter += 1
                current_state[current_state_counter] = obs.observation['player'][_ARMY_SUPPLY]
                current_state_counter += 1
                current_state[current_state_counter] = obs.observation['player'][_MINERALS]
                current_state_counter += 1
                #current_state[current_state_counter] = self.previous_supply_depot_during_build
                #current_state_counter += 1
                #current_state[current_state_counter] = self.previous_barracks_during_build
                #current_state_counter += 1
                #current_state[current_state_counter] = obs.observation['player'][_VESPENE]
                #current_state_counter += 1
                hot_squares = np.zeros(16)        
                enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
                for i in range(0, len(enemy_y)):
                    y = int(math.ceil((enemy_y[i] + 1) / 16))
                    x = int(math.ceil((enemy_x[i] + 1) / 16))
                    hot_squares[((y - 1) * 4) + (x - 1)] = 1
                
                if not self.base_top_left:
                    hot_squares = hot_squares[::-1]
                
                for i in range(0, 16):
                    current_state[i + current_state_counter] = hot_squares[i]

                if self.previous_action is not None:
                    self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
            
                #mcts_action = self.sendToDecisionMaker(current_state.tobytes())
                mcts_action = -1
                if(self.sentToDecisionMakerAsync == None or not self.sentToDecisionMakerAsync.isAlive()):
                    if(self.returned_action_from_decision_maker != -1):
                        mcts_action =  self.returned_action_from_decision_maker
                        print('mcts_action - {}'.format(mcts_action))
                        if(mcts_action >= 0 and mcts_action < 22):
                            print('mcts_action - {}'.format(smart_actions[mcts_action]))
                        else:
                            print('mcts_action - {}'.format(mcts_action))
                    else:
                        print('mcts = {}'.format(-1))
                    self.returned_action_from_decision_maker = -1

                    self.sentToDecisionMakerAsync = threading.Thread(target=self.sendToDecisionMaker)

                    self.current_state_for_decision_making = current_state

                    self.sentToDecisionMakerAsync.start()

                rl_action = self.qlearn.choose_action(str(current_state))

                # print('rl_action - {}'.format(smart_actions[rl_action]))

                self.previous_state = current_state
                self.previous_action = rl_action if mcts_action < 0  and mcts_action < 5 else mcts_action
            
                #print('state - {}    action - {}'.format(self.previous_state, self.previous_action))
                smart_action, x, y = self.splitAction(self.previous_action)
                
                if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                    unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                        
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]
                        
                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                    
                elif smart_action == ACTION_BUILD_MARINE:
                    if barracks_y.any():
                        i = random.randint(0, len(barracks_y) - 1)
                        target = [barracks_x[i], barracks_y[i]]
                
                        return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
                    
                elif smart_action == ACTION_ATTACK:
                    if _SELECT_ARMY in obs.observation['available_actions']:
                        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            
            elif self.move_number == 1:
                self.move_number += 1
                
                smart_action, x, y = self.splitAction(self.previous_action)
                    
                if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                    if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                        if self.cc_y.any():
                            if supply_depot_count == 0:
                                target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                            elif supply_depot_count == 1:
                                target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
                            self.previous_supply_depot_during_build += 1
                            return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
                
                elif smart_action == ACTION_BUILD_BARRACKS:
                    if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                        if self.cc_y.any():
                            if  barracks_count == 0:
                                target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                            elif  barracks_count == 1:
                                target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)
                            self.previous_barracks_during_build += 1
                            return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        
                elif smart_action == ACTION_BUILD_MARINE:
                    if _TRAIN_MARINE in obs.observation['available_actions']:
                        return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
            
                elif smart_action == ACTION_ATTACK:
                    do_it = True
                    
                    if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                        do_it = False
                    
                    if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                        do_it = False
                    
                    if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                        x_offset = random.randint(-1, 1)
                        y_offset = random.randint(-1, 1)
                        
                        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
                    
            elif self.move_number == 2:
                self.move_number = 0
                
                smart_action, x, y = self.splitAction(self.previous_action)
                    
                if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                    if _HARVEST_GATHER in obs.observation['available_actions']:
                        unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                        
                        if unit_y.any():
                            i = random.randint(0, len(unit_y) - 1)
                            
                            m_x = unit_x[i]
                            m_y = unit_y[i]
                            
                            target = [int(m_x), int(m_y)]
                            
                            return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
            
            return actions.FunctionCall(_NO_OP, [])
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logging.error(traceback.format_exc())
