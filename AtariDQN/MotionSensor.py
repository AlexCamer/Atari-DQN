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
        self.newState = np.stack([self.vecUpBound(movement), self.oldFrame]).astype(np.uint8).reshape((80,80,2))