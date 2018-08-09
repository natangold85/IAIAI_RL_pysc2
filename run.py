#!/usr/bin/python3

# run example: python .\run.py --map=Simple64 --typeFile=NaiveRunDiff2.txt --trainAgent=super --train=True
# kill al sc ps:  $ Taskkill /IM SC2_x64.exe /F

import os
import threading
import time
import tensorflow as tf
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


"""Script for starting all agents (a3c, very simple and slightly smarter).

This scripts is the starter for all agents, it has one command line parameter (--agent), that denotes which agent to run.
By default it runs the A3C agent.
"""

def run_thread(agent, display=False, difficulty = None):
    """Runs an agent thread.

    This helper function creates the evironment for an agent and starts the main loop of one thread.

    :param agent: agent to run
    :param ssize_x: X-size of the screen
    :param ssize_y: Y-size of the screen
    :param msize_x: X-size of the minimap
    :param msize_y: Y-size of the minimap
    :param display: whether to display the pygame output of an agent, for performance reasons deactivated by default
    """
    with sc2_env.SC2Env(map_name=flags.FLAGS.map,
                        agent_race='T',
                        bot_race='T',
                        difficulty=difficulty,
                        game_steps_per_episode=0,
                        screen_size_px=(SCREEN_SIZE, SCREEN_SIZE),
                        minimap_size_px=(MINIMAP_SIZE, MINIMAP_SIZE),
                        visualize=display) as env:
        run_loop.run_loop([agent], env)


def start_agent():
    """Starts the pysc2 agent.

    Helper function for setting up the A3C agents. If it is in training mode it starts PARALLEL_THREADS agents, otherwise
    it will only start one agent. It creates the TensorFlow session and TensorFlow's summary writer. If it is continuing
    a previous session and can't find an agent instance, it will just ignore this instance. It also initialises the
    weights of the neural network, if it doesn't find a previously saved one and initialises it. If it should show the
    pygame output of an agent, it only shows it for the first instance. Most of it's behaviour can be controlled with
    the same constants that can be found in a3c_agent.py and are also used by the A3C agent.
    """
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
