#!/usr/bin/python3

# run example: python .\run.py --map=Simple64 --typeFile=NaiveRunDiff2.txt --trainAgent=super --train=True
# kill al sc ps:  $ Taskkill /IM SC2_x64.exe /F

import os
import threading
import time
import tensorflow as tf
import collections
from absl import app
from absl import flags

from pysc2.env import run_loop
from pysc2.env import sc2_env

# all independent agents available
from agent_super import SuperAgent

from utils import SC2_Params


PARALLEL_THREADS = 16
RENDER = False
SCREEN_SIZE = SC2_Params.SCREEN_SIZE
MINIMAP_SIZE = SC2_Params.MINIMAP_SIZE

flags.DEFINE_string("trainAgent", "none", "Which agent to train.")
flags.DEFINE_string("play", "none", "Which agent to play.")
flags.DEFINE_string("typeFile", "none", "config file that builds heirarchi for decision maker (should contain a dict name dm_Types)")
flags.DEFINE_string("train", "True", "Which agent to train.")
flags.DEFINE_string("map", "none", "Which map to run.")

singlePlayerMaps = ["ArmyAttack5x5", "AttackBase"]

"""Script for starting all agents (a3c, very simple and slightly smarter).

This scripts is the starter for all agents, it has one command line parameter (--agent), that denotes which agent to run.
By default it runs the A3C agent.
"""

def run_thread(agent, display=False, difficulty = None):
    """Runs an agent thread."""
    players = [sc2_env.Agent(race=sc2_env.Race.terran)]
    if flags.FLAGS.map not in singlePlayerMaps:
        players.append(sc2_env.Bot(race=sc2_env.Race.terran, difficulty=difficulty))


    agent_interface_format=sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(screen=SCREEN_SIZE,minimap=MINIMAP_SIZE))

    with sc2_env.SC2Env(map_name=flags.FLAGS.map,
                        players=players,
                        game_steps_per_episode=0,
                        agent_interface_format=agent_interface_format,
                        visualize=display) as env:
        run_loop.run_loop([agent], env)


def start_agent():
    """Starts the pysc2 agent."""

    training = eval(flags.FLAGS.train)  
    if not training:
        parallel = 1
        show_render = True
    else:
        parallel = PARALLEL_THREADS
        show_render = RENDER

    # tables
    dm_Types = eval(open(flags.FLAGS.typeFile, "r+").read())

    if flags.FLAGS.trainAgent == "none":
        trainList = ["super"]
    else:
        trainList = flags.FLAGS.trainAgent
        trainList = trainList.split(",")

    if flags.FLAGS.play == "none":
        playList = trainList
    else:
        playList = flags.FLAGS.play
        playList = playList.split(",")

    print("\n\n\nplay list =", playList)        
    print("train list =", trainList, "\n\n\n")
    isMultiThreaded = parallel > 1

    decisionMaker = None
    agents = []
    for i in range(parallel):
        print("\n\n\n running thread #", i, "\n\n\n")
        agent = SuperAgent(decisionMaker=decisionMaker, isMultiThreaded=isMultiThreaded, dmTypes = dm_Types, playList=playList, trainList=trainList)
        
        if i == 0:
            decisionMaker = agent.GetDecisionMaker()

        agents.append(agent)

    threads = []
    show = show_render
    
    if "difficulty" in dm_Types:
        difficulty = int(dm_Types["difficulty"])
    else:
        print("run type not include difficulty\nExitting...")
        exit()

    for agent in agents:
        thread_args = (agent, show, difficulty)
        t = threading.Thread(target=run_thread, args=thread_args)
        threads.append(t)
        t.start()
        time.sleep(5)
        show = False

    for t in threads:
        t.join()


def main(argv):
    """Main function.

    This function check which agent was specified as command line parameter and launches it.

    :param argv: empty
    :return:
    """

    start_agent()


if __name__ == '__main__':
    print('Starting...')
    app.run(main)
