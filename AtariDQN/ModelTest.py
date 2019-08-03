import gym
import tensorflow as tf
from keras.models import load_model
import numpy as np
import collections
import gc
import cv2
import time
import matplotlib.pyplot as plt

env = gym.make('PongDeterministic-v0')
location = 'models/models/2019 07 31 00 15 58/model@410.h5'
model = tf.keras.models.load_model(location)
modRender = 1 
inputDim = (4, 80, 80)
actionSpace = env.action_space.n
inputFrames = 4

def getQs(state):
    a = model.predict(state)
    #print(a)
    return a

def step(action):
    futureState, reward, done, info = env.step(action)
    return preProcess(futureState), reward, done

def grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def downSample(frame):
    return cv2.resize(frame, (84, 110))[18:-8,:]

def preProcess(frame):
    return downSample(grayscale(frame))

def decayEpsilon(epsilon):
    return epsilon if epsilon <= minEpsilon else epsilon - decay

def save(episode):
    if episode%saveInterval is 0:
        agent.model.save(str(saveLocation) + '/model@' + str(episode) + '.h5')
        agent.model.save('models/recent.h5')
        stats.save(str(saveLocation))

currentDeque = collections.deque(maxlen=4)

while True:
    done = False
    epReward = 0
    currentFrame = preProcess(env.reset())
    for i in range(4): currentDeque.append(currentFrame)

    while not done:
        a = getQs(np.stack(currentDeque).reshape((1,84,84,4)))
        #print(a)
        if np.random.random() > 0.05: action = np.argmax(a)
        else: action = np.random.randint(0,actionSpace)
        newFrame, reward, done = step(action)
        #plt.imshow(currentDeque[0], cmap='gray')  # graph it
        #plt.show()
        currentDeque.append(newFrame)
        epReward += reward
        time.sleep(0.05)
        env.render()