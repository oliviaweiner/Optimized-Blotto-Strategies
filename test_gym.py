import gym
import numpy as np
import tensorflow as tf

import time
import random

RANDOM_SEED = 5

env = gym.make('CartPole-v1')
np.random.seed(5)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

print(env.action_space.shape)
print(env.observation_space.shape)