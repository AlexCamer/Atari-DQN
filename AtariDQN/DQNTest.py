import tensorflow as tf
import keras
import numpy as np
import gym
import collections
import random
import datetime
import os
import pickle
from Statistics import *

env = gym.make('CartPole-v0')
actionSpace = env.action_space.n
observationSpace = env.observation_space.shape[0]
targetUpdate = 5
time = datetime.datetime.now().strftime('%Y %m %d %H %M %S')
location = f'./models/{time}'
os.makedirs(location)
saveInterval = 100

alpha = 0.01
gamma = 0.99
epsilon = 1
decay = 0.99
maxMemory = 50000
minMemory = 1000
batchSize = 64
episodes = 1000
renderMod = 20

class DQNAgent:
    def __init__(self):
        self.model = self.createModel()
        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.model.get_weights())
        try:
            pickle_in = open('models/deque.pickle')
            self.memory = pickle.load(pickle_in)
        except:
            self.memory = collections.deque(maxlen = maxMemory)
        
        self.targetUpdateCounter = 0

    def createModel(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=observationSpace, activation='relu'))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(actionSpace, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=alpha))
        return model
    
    def memoryAppend(self, transition):
        self.memory.append(transition)

    def train(self, terminal):
        if len(self.memory) < minMemory:
            return

        batch = random.sample(self.memory, batchSize)
        
        currentStates = np.array([transition[0] for transition in batch])
        nextStates = np.array([transition[3] for transition in batch])

        currentQs_List = self.model.predict(currentStates)
        nextQs_List = self.model.predict(nextStates)
        
        X,y = [],[]
        for i, (currentState, action, reward, nextState, done) in enumerate(batch):
            currentQs = currentQs_List[i]
            currentQs[action] = reward if done else reward + gamma * np.max(nextQs_List[i])
            X.append(currentState)
            y.append(currentQs)

        self.model.fit(np.array(X), np.array(y), batch_size=batchSize, verbose=0)

        if terminal is True:
            self.targetUpdateCounter += 1

        if self.targetUpdateCounter >= targetUpdate:
            self.targetModel.set_weights(self.model.get_weights())
            self.targetUpdateCounter = 0

    def getQs(self, state):
        return self.model.predict(np.array(state))

agent = DQNAgent()
StatGraph = StatGraph(saveInterval)

for episode in range(1, episodes+1, 1):
    epReward = 0
    currentState = env.reset()
    done = False
    while not done:
        if np.random.random() > epsilon: action = np.argmax(agent.getQs(currentState.reshape((1,observationSpace))))
        else: action = np.random.randint(0, env.action_space.n)

        nextState, reward, done, info = env.step(action)

        #if episode%renderMod == 0: env.render()
        
        agent.memoryAppend((currentState, action, reward, nextState, done))
        epReward += reward
        agent.train(done)
        currentState = nextState

    print(episode)
    StatGraph.update(epReward)
    if epsilon > 0.001: epsilon *= decay

    if episode%saveInterval is 0:
        agent.model.save(f'{location}/model@{episode}.h5')
        StatGraph.save(location)
