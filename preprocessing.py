import numpy as np
from PIL import Image
import glob

def zero_padding(x):
    data = []
    for i in x:
        columns = i.shape[1]
        rows = i.shape[0]

        zeros = np.zeros((rows, 1))
        for j in range(int((64 - (columns%64))/2)):
            i = np.concatenate((i, zeros), axis=1)
            i = np.concatenate((zeros, i), axis=1)
        if columns%2 == 1:
            i = np.concatenate((i, zeros), axis=1)
        columns = i.shape[1]

        zeros = np.zeros((1, columns))
        for j in range(int((64 - (rows%64))/2)):
            i = np.concatenate((i, zeros), axis=0)
            i = np.concatenate((zeros, i), axis=0)
        if rows%2 == 1:
            i = np.concatenate((i, zeros), axis=0)
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

def delete_reshaping(x):
    data = []
    for i in x:
        columns = i.shape[1]
        rows = i.shape[0]
        delete_row = int((rows % 64)/2)
        delete_column = int((columns % 64)/2)
        if columns % 2 == 1 and rows % 2 == 1:
            i = i[delete_row+1:rows-delete_row,delete_column+1:columns-delete_column]
        elif rows % 2 == 1:
            i = i[delete_row+1:rows-delete_row,delete_column:columns-delete_column]
        elif columns % 2 == 1:
            i = i[delete_row:rows - delete_row, delete_column+1:columns - delete_column]
        else:
            i = i[delete_row:rows-delete_row,delete_column:columns-delete_column]

        data.append(i)

    return data


def mean_resizing(x,size):

    data = []
    for image in x:
        resizedData = []
        columns = image.shape[1]
        rows = image.shape[0]


        for i in range(0, rows, int(rows/size)):
            myRow = []
            for j in range(0, columns, int(columns/size)):
                myCell = (image[i:i + int(rows/size), j:j + int(columns/size)])
                max = np.mean(myCell)
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


x = delete_reshaping(x)
x = mean_resizing(x,64)

x = np.array(x)

im = Image.fromarray(x[5])
im.show()
