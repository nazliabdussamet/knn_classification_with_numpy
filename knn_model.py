import numpy as np

data = np.load("data.npy")

np.random.shuffle(data)

x = data[:,1:4097]
y = data[:,-1:]




