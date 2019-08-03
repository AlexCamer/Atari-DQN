import gym
import collections
import matplotlib.pyplot as plt
import numpy as np
import cv2

env = gym.make('Pong-v0')

def step(action):
    futureState, reward, done, info = env.step(action)
    return preProcess(futureState), reward, done

def grayscale(frame):
    return np.mean(frame, axis=2)

def downSample(frame):
    return cv2.resize(frame, (84, 110))[18:-8,:]

def preProcess(frame):
    return downSample(grayscale(frame))

def LoBound(x):
    return x if x>=0 else 0

def UpBound(x):
    return x if x<=255 else 255

class MotionSensor:
    def __init__(self):
        self.oldFrame = np.zeros((80,80))
        self.newState = np.zeros((80,80,2))
        self.deque = collections.deque(maxlen=3)
        for i in range(3): self.deque.appendleft(self.oldFrame)
        self.vecLoBound = np.vectorize(LoBound)
        self.vecUpBound = np.vectorize(UpBound)
    def updateState(self, newFrame):
        motion = self.vecLoBound(newFrame - self.oldFrame)
        self.deque.appendleft(motion)
        self.oldFrame = newFrame
        self.compile()
    def compile(self):
        movement = self.deque[0] + 0.6*self.deque[1] +  0.3*self.deque[2] 
        self.oldState = self.newState
        self.newState = self.vecUpBound(movement).astype(np.uint8).reshape((80,80))

done = False
#motionSensor = MotionSensor()
#motionSensor.updateState(preProcess())
env.reset()

for i in range(500):
    action = np.random.randint(0, 6)
    newFrame, reward, done = step(action)
    #motionSensor.updateState(newFrame)
    plt.imshow(newFrame, cmap='gray')  # graph it
    #plt.show()
    plt.draw()
    plt.pause(0.0001)
    env.render()
    print(newFrame.shape)




    