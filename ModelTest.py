import gym
import tensorflow as tf
from keras.models import load_model
import numpy as np
import collections
import gc
import cv2
import time
import matplotlib.pyplot as plt
from StateProcessor import *

env = gym.make('BreakoutDeterministic-v4')
location = 'models/recent.h5'
#location = 'models/2019 07 31 04 37 03/model@610.h5'
model = tf.keras.models.load_model(location)
actionSpace = env.action_space.n

def LoBound(x):
    return x if x>=0 else 0

def UpBound(x):
    return x if x<=255 else 255

def step(action):
    futureState, reward, done, info = env.step(action)
    return preProcess(futureState), reward, done

def grayscale(frame):
    return np.mean(frame, axis=2)

def downSample(frame):
    return cv2.resize(frame, (94, 110))[18:-8,5:-5]

def preProcess(frame):
    return downSample(grayscale(frame))

def getQs(state):
        return model.predict(state/255)

stateProcessor = StateProcessor()

while True:
    done = False
    stateProcessor.update(preProcess(env.reset()))
    epReward = 0

    while not done:
        if np.random.random() > 0.02: action = np.argmax(getQs(stateProcessor.newState.reshape((1,84,84,1))))
        else: action = np.random.randint(0,actionSpace)
        newFrame, reward, done = step(action)
        stateProcessor.update(newFrame)
        env.render()
        epReward += reward
        time.sleep(0.03)
    print(epReward)