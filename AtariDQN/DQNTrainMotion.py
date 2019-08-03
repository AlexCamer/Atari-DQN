import tensorflow as tf
import keras
import numpy as np
import cv2
import random
import datetime
import gym
import collections
import os
from Statistics import *

alpha = 0.00001
epsilon = 1
minEpsilon = 0.05
decay = (epsilon - minEpsilon)/2500
gamma = 0.99
episodes = 12000
maxMemory = 500000
minMemory = 50000
batchSize = 32
saveInterval = 50

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

env = gym.make('BreakoutDeterministic-v4')
actionSpace = env.action_space.n
time = datetime.datetime.now().strftime('%Y %m %d %H %M %S')
saveLocation = './models/' + str(time)
os.makedirs(saveLocation)

class DQNAgent:
    def __init__(self):
        self.memory = collections.deque(maxlen = maxMemory)
        #self.model = tf.keras.models.load_model('models/recent.h5')
        self.model = self.createModel()
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())
    
    def createModel(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), padding='same', activation='relu', input_shape=(84,84,2)))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(actionSpace, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))
        return model

    def train(self, terminal):
        if len(self.memory) < minMemory: return
        batch = random.sample(self.memory, batchSize)
        currentState_List = np.array([transition[0] for transition in batch])
        futureState_List = np.array([transition[3] for transition in batch])
        currentQs_List = self.model.predict(currentState_List)
        futureQs_List = self.targetModel.predict(futureState_List)

        X,y = [],[] 
        for i, (currentState, action, reward, futureState, done) in enumerate(batch):
            currentQs = currentQs_List[i]
            currentQs[action] = reward if done else reward + gamma * np.max(futureQs_List[i])
            X.append(currentState)
            y.append(currentQs)

        self.model.fit(np.array(X), np.array(y), verbose=0)
        if terminal: self.targetModel.set_weights(self.model.get_weights())

    def getQs(self, state):
        return self.model.predict(state)

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
        #plt.pause(0.0001)
        self.newState = np.stack([self.vecUpBound(movement), self.oldFrame]).astype(np.uint8).reshape((84,84,2))

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

def decayEpsilon(epsilon):
    return epsilon if epsilon <= minEpsilon else epsilon - decay

def save(episode):
    if episode%saveInterval is 0:
        agent.model.save(str(saveLocation) + '/model@' + str(episode) + '.h5')
        agent.model.save('models/recent.h5')
        stats.save(str(saveLocation))

agent = DQNAgent()
stats = StatGraph(saveInterval)
motionSensor = MotionSensor()

for episode in range(1, episodes+1, 1):
    done = False
    epReward = 0
    motionSensor.update(preProcess(env.reset()))

    while not done:
        if np.random.random() > epsilon: action = np.argmax(agent.getQs(motionSensor.newState.reshape((1,84,84,2))))
        else: action = np.random.randint(0,actionSpace)
        newFrame, reward, done = step(action)
        motionSensor.update(newFrame)
        agent.memory.append((motionSensor.oldState, action, reward, motionSensor.newState, done))
        epReward += reward
        agent.train(done)
    
    print(episode, np.round(epsilon,2), epReward, len(agent.memory))
    stats.update(epReward)
    epsilon = decayEpsilon(epsilon)
    save(episode)