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

    neighbors = []

    for i in range(num_neighbors):
        neighbors.append(distances[i][1])
    return neighbors

def find_class(train, test_row, num_neighbors):

    neighbors = np.array(find_neighbors(train, test_row, num_neighbors))
    ones = np.count_nonzero(neighbors == 1)

    if ones > int((len(neighbors) - 1)  / 2):
        return 1
    else:
        return 0

data = np.load("data.npy")

np.random.shuffle(data)


data_train = data[:450,:]
data_test = data[450:500,:]




true_predictions = []
num_neighbors = []

for k in range(1,30,2):

    predicted_values = []

    for test_row in data_test:
        prediction = find_class(data_train, test_row, k)
        predicted_values.append(prediction)

    true_prediction = 0

    for i in range(50):
        if data_test[i,-1] == predicted_values[i]:
            true_prediction += 1
        else:
            continue

    true_predictions.append(true_prediction)
    num_neighbors.append(k)

plt.plot(num_neighbors,true_predictions)
plt.show()


