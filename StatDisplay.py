import pickle

location = 'models/RecentGraphs.pickle'
#location = 'models/2019 08 10 13 00 00/graphs.pickle'
#location = 'OptimalEpsilon/graphs.pickle'
 
pickle_in = open(location, 'rb')
graphs = pickle.load(pickle_in)

graphs.display()