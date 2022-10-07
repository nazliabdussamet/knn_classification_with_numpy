import numpy as np

data = np.load("data.npy")

np.random.shuffle(data)

x = data[:,1:4097]
y = data[:,-1:]

x_train = x[:450,:]
x_test = x[450:500,:]
y_train = y[:450,:]
y_test = y[450:500,:]




