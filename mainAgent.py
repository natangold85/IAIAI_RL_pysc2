#!/usr/bin/python3

import os
import threading
import time
import tensorflow as tf
from absl import app
from absl import flags
from pysc2.env import run_loop
from pysc2.env import sc2_env

# all independent agents available
from resource_gather import Gather # map: CollectMineralsAndGas

PARALLEL_THREADS = 16
RENDER = False
TRAINING = True

flags.DEFINE_string("agent", "none", "Which agent to run.")
flags.DEFINE_string("map", "none", "Which map to run.")
flags.DEFINE_string("type", "play", "Which type of decisionMaker to run.")


"""Script for starting all agents (a3c, very simple and slightly smarter).

This scripts is the starter for all agents, it has one command line parameter (--agent), that denotes which agent to run.
By default it runs the A3C agent.
"""

def run_thread(agent, display=False):
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
                        game_steps_per_episode=0,
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

    if not TRAINING:
        parallel = 1
        show_render = True
    else:
        parallel = PARALLEL_THREADS
        show_render = RENDER

    agents = []
    # tables

    agentClass = eval(flags.FLAGS.agent)
    decisionMaker = None
    isMultiThreaded = parallel > 1

    for i in range(parallel):
        print("\n\n\n running thread #", i, "\n\n\n")
        agent = agentClass(runArg=flags.FLAGS.type, decisionMaker=decisionMaker, isMultiThreaded=isMultiThreaded)
        
        if i == 0:
            decisionMaker = agent.GetDecisionMaker()

        agents.append(agent)

    threads = []
    show = show_render
    for agent in agents:
        thread_args = (agent, show)
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
