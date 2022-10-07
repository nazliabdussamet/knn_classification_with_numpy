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


def mean_resizing(x,y,size):

    data = []
    k = 0
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

        resizedData = [item for sublist in resizedData for item in sublist]
        resizedData.append(y[k])
        k += 1
        data.append(resizedData)

    return data


x = []
y = []

pathCar = "C:/python_projects/knn_classification_with_numpy/cars/*"
pathPlane = "C:/python_projects/knn_classification_with_numpy/planes/*"
path1 = glob.glob(pathCar)
path2 = glob.glob(pathPlane)


for file in path1:
    image = np.array(Image.open(file))
    if image.ndim < 3:
        continue
    else:
        image = np.add(image[:, :, 0] * 0.2989, image[:, :, 1] * 0.5870, image[:, :, 2] * 0.1140)
    x.append(image)
    y.append(int(0))


for file in path2:
    image = np.array(Image.open(file))
    if image.ndim < 3:
        continue
    else:
        image = np.add(image[:,:,0]*0.2989, image[:,:,1]*0.5870,image[:,:,2]*0.1140)
    x.append(image)
    y.append(int(1))


x = delete_reshaping(x)
data = mean_resizing(x,y,64)

np.save("data",data)