import tensorflow as tf
import keras
import numpy as np
import cv2
import random
import datetime
import gym
import collections
import os
from AgentMemory import *
from Statistics import *
from FrameProcessor import *

alpha = 0.0000625
epsilon = 1
minEpsilon = 0.07
decay = (epsilon - minEpsilon)/1000
gamma = 0.99
episodes = 12000
maxMemory = 750000
minMemory = 10000
batchSize = 32
maxPriorityMemory = 10
priorityBatchSize = 4
priorityQualify = 1.5
saveInterval = 50
movingAvgLen = 50
minScore = 0
targetUpdate = 10000

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

env = gym.make('BreakoutDeterministic-v4')
actionSpace = env.action_space.n
time = datetime.datetime.now().strftime('%Y %m %d %H %M %S')
saveLocation = './models/' + str(time)
os.makedirs(saveLocation)

class DQNAgent:
    def __init__(self):
        self.memory = collections.deque(maxlen=maxMemory) #RandomReplaceMemory(maxlen=maxMemory)
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

    def train(self, terminal):
        if len(self.memory) <= minMemory: return
        batch = random.sample(self.memory, batchSize) + self.priorityMemory.randomSample(priorityBatchSize)
        currentState_List = np.array([transition[0] for transition in batch])
        futureState_List = np.array([transition[3] for transition in batch])
        currentQs_List = self.model.predict(currentState_List/255)
        futureQs_List = self.targetModel.predict(futureState_List/255)

        X,y = [],[] 
        for i, (currentState, action, reward, futureState, done) in enumerate(batch):
            currentQs = currentQs_List[i]
            currentQs[action] = reward if done else reward + gamma * np.max(futureQs_List[i])
            X.append(currentState)
            y.append(currentQs)

        self.model.fit(np.array(X)/255, np.array(y), verbose=0)
        if self.frames%targetUpdate is 0: self.targetModel.set_weights(self.model.get_weights())

    def getQs(self, state):
        return self.model.predict(state/255)

    def memAppend(self, transition, epReward):
        self.frames += 1
        self.memory.append(transition)
        self.recentRun.append(transition)
        if transition[4] is True:
            self.scoreHistory.append(epReward)
            self.priorityMemory.decay()
            epScore = (epReward - minScore)/(np.average(self.scoreHistory)-minScore)
            if len(self.scoreHistory) >= movingAvgLen and epScore >= priorityQualify: self.priorityMemory.add(self.recentRun, int(epScore*10))
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

agent = DQNAgent()
stats = StatGraph(saveInterval)
frameProcessor = FrameProcessor()

for episode in range(1, episodes+1, 1):
    done = False
    epReward = 0
    frameProcessor.update(preProcess(env.reset()))

    while not done:
        if np.random.random() > epsilon: action = np.argmax(agent.getQs(frameProcessor.newState.reshape((1,84,84,1))))
        else: action = np.random.randint(0,actionSpace)
        newFrame, reward, done = step(action)
        epReward += reward
        frameProcessor.update(newFrame)
        agent.memAppend((frameProcessor.oldState, action, reward, frameProcessor.newState, done), epReward)
        agent.train(done)
    
    print(episode, np.round(epsilon,2), epReward, agent.frames, len(agent.memory), len(agent.priorityMemory.list))
    stats.update(epReward)
    epsilon = processEpsilon(epsilon)
    save(episode)