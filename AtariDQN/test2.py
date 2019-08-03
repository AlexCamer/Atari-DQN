import numpy as np
import sys
import pickle

a = np.zeros((1000,1000))
print(a.astype(np.uint64).nbytes)

pickle_out = open('test', 'wb')
pickle.dump(a, pickle_out)
pickle_out.close()