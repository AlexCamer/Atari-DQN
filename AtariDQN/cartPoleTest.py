import gym
import tensorflow as tf
from keras.models import load_model
import numpy as np

env = gym.make('CartPole-v0')
location = 'models/recent.h5'
model = tf.keras.models.load_model(location)
print(env.observation_space.shape[0])

def getQs(state):
    return model.predict(np.array(state))

while True:
    done = False
    epReward = 0
    currentState = env.reset()
    while not done:
        a = getQs(currentState.reshape((1,4)))
        print(a)
        action = np.argmax(a)
        nextState, reward, done, info = env.step(action)
        env.render()
        epReward += reward
        currentState = nextState
    print(epReward)