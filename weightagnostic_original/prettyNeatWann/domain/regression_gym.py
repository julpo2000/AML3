import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2
import math
import tensorflow as tf
from skimage import color

class RegressionEnv(gym.Env):
  """Classification as an unsupervised OpenAI Gym RL problem.
  Includes scikit-learn digits dataset, MNIST dataset
  """

  def __init__(self, trainSet, target):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you need them
    self.batch   = 1000 # Number of images per batch
    self.seed()
    self.viewer = None

    self.trainSet = trainSet
    self.target   = target

    nInputs = np.shape(trainSet)[1]
    high = np.array([1.0]*nInputs)
    self.action_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))

    self.state = None
    self.trainOrder = None
    self.currIndx = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.trainOrder = np.random.permutation(len(self.target))
    self.t = 0 # timestep
    self.currIndx = self.trainOrder[self.t:self.t+self.batch]
    self.state = self.trainSet[self.currIndx,:]
    return self.state
  
  def step(self, action):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    y = self.target[self.currIndx]

    # print("action:")
    # print(action)

    # print("y:")
    # print(y)

    # print(action - y)

    mse_action = (np.square(action)).mean(axis=None)

    # if mse_action != 0:
    #   print("mse action:")
    #   print(mse_action)
    #   print("mse y:")
    #   print((np.square(y)).mean(axis=None))
    loss = np.sqrt((np.square(action - y)).mean(axis=None))

    reward = -loss

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True
      self.currIndx = self.trainOrder[(self.t*self.batch):\
                                      (self.t*self.batch + self.batch)]

      self.state = self.trainSet[self.currIndx,:]
    else:
      done = True

    obs = self.state
    return obs, reward, done, {}


# -- Data Sets ----------------------------------------------------------- -- #

def sine():
  ''' 
  Generates a single sine wave
  '''  

  x_train = np.arange(0, 1*3.14, 0.1)
  y_train = np.sin(x_train)

  y_train = np.reshape(y_train, (len(y_train)))
  x_train = x_train.reshape(-1, (1))
  return x_train, y_train


def addition():
  ''' 
  Adds two random ints
  '''  

  x_train = np.random.randint(0, 100, (10000, 2))
  y_train = np.sum(x_train, axis=1)
  print(x_train)
  print(y_train)

  y_train = np.reshape(y_train, (len(y_train)))
  x_train = x_train.reshape(-1, (1))
  return x_train, y_train



 
