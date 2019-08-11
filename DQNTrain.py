import tensorflow as tf
import keras
import numpy as np
import cv2
import random
import datetime
import gym
import collections
import os
import time
from AgentMemory import *
from Statistics import *
from FrameProcessor import *

alpha = 0.00001
epsilon = 1
minEpsilon = 0.07
decay = (epsilon - minEpsilon)/200
gamma = 0.99
episodes = 12000
maxMemory = 40000
minMemory = 10000
batchSize = 32
maxPriorityMemory = 10
priorityBatchSize = 4
priorityQualify = 1.5
saveInterval = 50
movingAvgLen = 50
minScore = -21
targetUpdate = 10000

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

env = gym.make('BreakoutDeterministic-v4')
actionSpace = env.action_space.n
Time = datetime.datetime.now().strftime('%Y %m %d %H %M %S')
saveLocation = './models/' + str(Time)
os.makedirs(saveLocation)

class DQNAgent:
    def __init__(self):
        self.memory = RandomReplaceMemory(maxlen=maxMemory) #collections.deque(maxlen=maxMemory)
        self.priorityMemory = PriorityMemory(maxlen=maxPriorityMemory)
        self.scoreHistory = collections.deque(maxlen=movingAvgLen)
        self.recentRun = []
        #self.model = tf.keras.models.load_model('models/recent.h5')
        self.model = self.createModel()
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())
        self.frames = 0
    
    def createModel(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), padding='same', activation='relu', input_shape=(84,84,1)))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(actionSpace, activation='linear'))
        model.compile(loss=tf.compat.v1.losses.huber_loss, optimizer=keras.optimizers.Adam(lr=alpha))
        return model

    def train(self, done):
        if len(self.memory.list) <= minMemory: return
        batch = self.memory.randomSample(batchSize) + self.priorityMemory.randomSample(priorityBatchSize)
        currentState_List = np.array([transition[0] for transition in batch])
        futureState_List = np.array([transition[3] for transition in batch])
        currentQs_List = self.model.predict(currentState_List/255)
        futureQs_List = self.targetModel.predict(futureState_List/255)

        X,y = [],[] 
        for i, (currentState, action, reward, futureState, terminal) in enumerate(batch):
            currentQs = currentQs_List[i]
            currentQs[action] = reward if terminal else reward + gamma * np.max(futureQs_List[i])
            X.append(currentState)
            y.append(currentQs)

        self.model.fit(np.array(X)/255, np.array(y), verbose=0)
        if done: self.targetModel.set_weights(self.model.get_weights())

    def getQs(self, state):
        return self.model.predict(state/255)

    def memAppend(self, transition, done, epReward):
        self.frames += 1
        self.memory.add(transition)
        self.recentRun.append(transition)
        if done:
            self.scoreHistory.append(epReward)
            self.priorityMemory.decay()
            epScore = (epReward - minScore)/(np.average(self.scoreHistory)-minScore)
            if epScore >= priorityQualify: self.priorityMemory.add(self.recentRun, int(epScore*10))
            self.recentRun = []

def step(action):
    futureState, reward, done, info = env.step(action)
    return preProcess(futureState), reward, done

def grayscale(frame):
    return np.mean(frame, axis=2)

def downSample(frame):
    return cv2.resize(frame, (94, 110))[18:-8,5:-5]

def preProcess(frame):
    return downSample(grayscale(frame))

def processEpsilon(epsilon):
    if epsilon <= minEpsilon:
        return random.uniform(0.005, minEpsilon)
    else:
        return epsilon - decay

def save(episode):
    if episode%saveInterval is 0:
        agent.model.save(str(saveLocation) + '/model@' + str(episode) + '.h5')
        agent.model.save('models/recent.h5')
        stats.save(str(saveLocation))

def isTerminal(lives):
    currentLives = env.ale.lives()
    terminal = False
    if currentLives != lives:
        lives = currentLives
        terminal = True
    return lives, terminal

agent = DQNAgent()
stats = StatGraph(saveInterval)
frameProcessor = FrameProcessor()

for episode in range(1, episodes+1, 1):
    done = False
    epReward = 0
    lives = env.ale.lives()
    frameProcessor.update(preProcess(env.reset()))

    while not done:
        if np.random.random() > epsilon: action = np.argmax(agent.getQs(frameProcessor.newState.reshape((1,84,84,1))))
        else: action = np.random.randint(0,actionSpace)
        newFrame, reward, done = step(action)
        lives,terminal = isTerminal(lives)
        epReward += reward
        frameProcessor.update(newFrame)
        agent.memAppend((frameProcessor.oldState, action, reward, frameProcessor.newState, terminal), done, epReward) #terminal = live lost, done = all lives lost
        agent.train(done)

    print(episode, np.round(epsilon,2), epReward, agent.frames, len(agent.memory.list), len(agent.priorityMemory.list))
    stats.update(epReward)
    epsilon = processEpsilon(epsilon)
    save(episode)