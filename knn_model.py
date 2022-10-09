import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)

def find_neighbors(train, test_row, num_neighbors):

    distances = []

    for train_row in train:
        dist = euclidean_distance(train_row, test_row)
        distances.append((dist,train_row[-1]))

    distances.sort(key=lambda tup: tup[0])

    neighbors_list = []
    for j in num_neighbors:
        neighbors = []
        for i in range(j):
            neighbors.append(distances[i][1])
        neighbors_list.append(neighbors)

    return neighbors_list

def find_class(train, test_row, num_neighbors):

    neighbors_list = find_neighbors(train, test_row, num_neighbors)

    predictions = []
    for i in range(len(neighbors_list)):
        my_list2 = np.array(neighbors_list[i])
        ones = np.sum(my_list2)

        if ones > int((len(neighbors_list[i]) - 1)  / 2):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

data = np.load("data2.npy")

np.random.shuffle(data)

slicer = int(data.shape[0]*0.9)
data_train = data[:slicer,:]
data_test = data[slicer:data.shape[0],:]


predicted_values = []
my_list = range(1,60,2)
for test_row in data_test:
    predictions = find_class(data_train, test_row, my_list)
    predicted_values.append(predictions)

k = []
for i in range(len(predicted_values[0])):

    true_prediction = 0
    for j in range(len(data_test)):
        if data_test[j,-1] == predicted_values[j][i]:
            true_prediction += 1
        else:
            continue

    k.append(true_prediction/(len(data_test)+1))

plt.plot(my_list,k)
plt.show()


