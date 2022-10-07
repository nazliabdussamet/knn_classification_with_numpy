import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

def zero_padding(x):
    data = []
    for i in x:
        columns = i.shape[1]
        rows = i.shape[0]

        zeros = np.zeros((rows, 1))
        for j in range(int((64 - (columns%64))/2)):
            i = np.concatenate((i, zeros), axis=1)
            i = np.concatenate((zeros, i), axis=1)
            columns = i.shape[1]

        zeros = np.zeros((1, columns))
        for j in range(int((64 - (rows%64))/2)):
            i = np.concatenate((i, zeros), axis=0)
            i = np.concatenate((zeros, i), axis=0)
        data.append(i)

    return data

def squared_zero_padding(x):
    data = []
    for i in x:
        columns = i.shape[1]
        rows = i.shape[0]
        if columns > rows:
            padding = columns + int((64 - (columns%64)))
        elif rows > columns:
            padding = rows + int((64 - (rows%64)))
        else:
            padding = columns + int((64 - (columns % 64)))

        zeros = np.zeros((rows, 1))
        for j in range(int((padding - columns) / 2)):
            i = np.concatenate((i, zeros), axis=1)
            i = np.concatenate((zeros, i), axis=1)
            columns = i.shape[1]

        zeros = np.zeros((1, columns))
        for j in range(int((padding - rows) / 2)):
            i = np.concatenate((i, zeros), axis=0)
            i = np.concatenate((zeros, i), axis=0)
        print(i.shape)
        data.append(i)

    return data

def max_resizing(x):

    data = []

    for image in x:
        resizedData = []
        columns = image.shape[1]
        rows = image.shape[0]


        for i in range(0, rows, int(rows/64)):
            myRow = []
            for j in range(0, columns, int(columns/64)):
                myCell = (image[i:i + int(rows/64), j:j + int(columns/64)])
                max = np.max(myCell)
                myRow.append(max)

            resizedData.append(myRow)
        data.append(resizedData)

    return data


x = []
y = []

pathString = "C:/python_projects/knn_classification_with_numpy/test_data/*.png"
path = glob.glob(pathString)

for file in path:
    image = Image.open(file)
    image = image.convert("L")
    data = np.array(image)
    x.append(data)

x = np.array(x)
y = np.array(y)

x = squared_zero_padding(x)
x = max_resizing(x)

plt.imshow(x[0])
plt.show()
