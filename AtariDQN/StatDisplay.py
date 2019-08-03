import pickle

location = 'models/RecentGraphs.pickle'
 
pickle_in = open(location, 'rb')
graphs = pickle.load(pickle_in)

graphs.display()