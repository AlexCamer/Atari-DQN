import gym
import random
import numpy as np
import time

env = gym.make('PongDeterministic-v4')
actionSpace = env.action_space.n

while True:
    env.reset()
    done = False
    while not done:
        a,b,done,c = env.step(np.random.randint(0,actionSpace))
        print(env.ale.lives())
        env.render()
        #time.sleep(0.05)