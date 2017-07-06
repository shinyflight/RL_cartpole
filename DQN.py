import gym
import os
# control tf message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import numpy as np
import yaml

# yaml hyperparameter integration
with open("DQN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = config['GPU']

# Agent
class QAgent:
    # TODO: neural network for Q-Agent

    # select action based on action-value(Q) + epsilon-greedy
    def sample_action(self, Q, feed, eps, options):
        # estimate action value function
        act_values = Q.eval(feed)
        # epsilon-greedy policy : exploration
        if random.random() <=eps:
            action_index = random.randrange(config['ACTION_DIM'])
        # limit of value-based method (choose a action that have maximum value)
        # : slight change of value can make big change of policy
        else:
            action_index = np.argmax(act_values)
