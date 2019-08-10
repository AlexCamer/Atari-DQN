import numpy as np
from AgentMemory import *

a = []

while True:
    a.append(np.zeros((100,100)))
    if len(a) >= 10000: a = []