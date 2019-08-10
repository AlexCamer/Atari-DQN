import numpy as np
import collections

class FrameProcessor:
    def __init__(self):
        self.oldFrame = np.zeros((84,84))
        self.newState = np.zeros((84,84,1))
        self.deque = collections.deque(maxlen=5)
        for i in range(5): self.deque.append(self.oldFrame)
        self.vecLoBound = np.vectorize(LoBound)
    def update(self, newFrame):
        motion = self.vecLoBound(newFrame - self.oldFrame)
        motion[81:,:] = 0
        self.oldFrame = newFrame
        self.deque.append(motion)
        self.compile()
    def compile(self):
        movement = self.deque[4] + 0.8*self.deque[3] + 0.6*self.deque[2] + 0.4*self.deque[1] + 0.2*self.deque[0]
        self.oldState = self.newState
        self.newState = np.mean(np.stack([movement,self.oldFrame]), axis=0).astype(np.uint8).reshape((84,84,1))

def LoBound(x):
    return x if x>=0 else 0