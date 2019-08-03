import gym
import tensorflow as tf
from keras.models import load_model
import numpy as np
import collections
import gc
import cv2
import time
import matplotlib.pyplot as plt

env = gym.make('BreakoutDeterministic-v4')
location = 'models/recent.h5'
#location = 'models/2019 07 31 04 37 03/model@610.h5'
model = tf.keras.models.load_model(location)
actionSpace = env.action_space.n

class MotionSensor:
    def __init__(self):
        self.oldFrame = np.zeros((84,84))
        self.newState = np.zeros((84,84,2))
        self.deque = collections.deque(maxlen=3)
        for i in range(3): self.deque.append(self.oldFrame)
        self.vecLoBound = np.vectorize(LoBound)
        self.vecUpBound = np.vectorize(UpBound)
    def update(self, newFrame):
        motion = self.vecLoBound(newFrame - self.oldFrame)
        self.deque.append(motion)
        self.oldFrame = newFrame
        self.compile()
    def compile(self):
        movement = self.deque[2] + 0.7 * self.deque[1] + 0.4 * self.deque[0]
        self.oldState = self.newState
        #plt.imshow(movement, cmap='gray')  # graph it
        #plt.draw()
        plt.pause(0.0001)
        self.newState = np.stack([self.vecUpBound(movement), self.oldFrame]).astype(np.uint8).reshape((84,84,2))

def getQs(state):
        return model.predict(state)

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
    return cv2.resize(frame, (84, 110))[18:-8,:]

def preProcess(frame):
    return downSample(grayscale(frame))

motionSensor = MotionSensor()

while True:
    done = False
    motionSensor.update(preProcess(env.reset()))

    while not done:
        if np.random.random() > 0.05: action = np.argmax(getQs(motionSensor.newState.reshape((1,84,84,2))))
        else: action = np.random.randint(0,actionSpace)
        newFrame, reward, done = step(action)
        motionSensor.update(newFrame)
        env.render()
        time.sleep(0.03)
