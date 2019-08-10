import matplotlib.pyplot as plt
import pickle

plt.style.use('ggplot')

class StatParent:
    def __init__(self):
        self.minimum, self.maximum, self.mean, self.x = [], [], [], []

    def display(self):
        plt.plot(self.x, self.minimum)
        plt.plot(self.x, self.maximum)
        plt.plot(self.x, self.mean)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.show()

class StatGraph(StatParent):
    def __init__(self, chunkSize=100):
        self.memory = []
        self.chunkSize = chunkSize
        self.chunk = -1
        self.size = 0

    def update(self, reward):
        if self.size%self.chunkSize == 0:
            self.chunk += 1
            self.memory.append([])
        self.memory[self.chunk].append(reward)
        self.size += 1

    def save(self, location):
        self.process()
        info = statStorage(self.minimum, self.maximum, self.mean, self.x)
        pickle_out = open(str(location) + '/graphs.pickle', 'wb')
        pickle.dump(info, pickle_out)
        pickle_out.close()
        pickle_out = open('models/RecentGraphs.pickle', 'wb')
        pickle.dump(info, pickle_out)
        pickle_out.close()

    def process(self):
        self.minimum, self.maximum, self.mean = [], [], []
        for chunk in self.memory:
            self.minimum.append(min(chunk))
            self.maximum.append(max(chunk))
            self.mean.append(sum(chunk)/len(chunk)) 
        chunkAmount = len(self.minimum)
        self.x = [i*self.chunkSize for i in list(range(1, chunkAmount+1, 1))]

class statStorage(StatParent):
    def __init__(self, minimum, maximum, mean, x):
        self.minimum=minimum
        self.maximum=maximum
        self.mean=mean
        self.x=x
