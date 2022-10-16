import numpy as np
import scipy.special as spl
import matplotlib.pyplot as plt

data_num = [10,20,100,1000,10000]
train_rmse_list = []
test_rmse_list = []


for i in data_num:
    x = np.linspace(0,8,int(i))
    y = spl.jv(0,x)
    y_noise = y + (np.random.normal(0,1,len(y)) * 0.1)

    x_train = x[0:int(i*0.8)]
    x_test = x[int(i*0.8):int(i)]
    y_train = y_noise[0:int(i*0.8)]
    y_test = y_noise[int(i*0.8):int(i)]

    curve = np.regpolyfit(x_train, y_train, 8,0)
    poly = np.poly1d(curve)

    x_predicted = []
    y_predicted = []

    for i in x:
        x_predicted.append(i)
        y_predicted.append(poly(i))


    train_rmse = 0
    for i in range(0,len(x_train)):
        rmse = (y_predicted[i] - y_train[i])**2
        train_rmse += rmse
    train_rmse = np.sqrt(train_rmse/len(x_train))
    train_rmse_list.append(train_rmse)

    test_rmse =  0
    for i in range(0,len(x_test)):
        rmse = (y_predicted[len(x_train)+i] - y_test[i])**2
        test_rmse += rmse

    test_rmse = np.sqrt(test_rmse / len(x_test))
    test_rmse_list.append(test_rmse)

    plt.scatter(x,y_noise)
    plt.plot(x_predicted,y_predicted)
    plt.show()

print(test_rmse_list)
print(train_rmse_list)