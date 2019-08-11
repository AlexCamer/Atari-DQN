import collections
import random

class PriorityMemory:
  def __init__(self, maxlen):
    self.list = [] #collections.deque(maxlen=maxlen)
    self.maxlen = maxlen
  def randomSample(self, size):
    if len(self.list) > 0: return random.sample(self.list[random.randint(0, len(self.list)-1)].content, size)
    return []
  def decay(self):
    for i, chunk in enumerate(self.list):
      chunk.countdown -= 1 
      if chunk.countdown <= 0: del self.list[i]
  def add(self, content, countdown):
    if len(self.list) >= self.maxlen: self.list.pop(0)
    self.list.append(PriorityMemoryChunk(content, countdown))

class PriorityMemoryChunk:
  def __init__(self, content, countdown):
    self.content = content
    self.countdown = countdown

class RandomReplaceMemory:
  def __init__(self, maxlen):
    self.list = []
    self.maxlen = maxlen
  def add(self, element):
    if len(self.list) >= self.maxlen: self.list[random.randint(0, self.maxlen-1)] = element
    else: self.list.append(element)
  def randomSample(self, size):
    return random.sample(self.list, size)    