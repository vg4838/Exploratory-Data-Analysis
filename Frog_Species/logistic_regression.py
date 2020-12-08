import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# parameters for frogs complete dataset are [[  0.18149773] [-13.71726613] [ -8.92281811]]
# parameters for frogs complete dataset are [[-0.43699085] [-19.69328725] [-14.50497004]]

def calculate_cost(X, y, theta):
    h_theta_x = 1 / (1 + np.exp(np.matmul(-X, theta)))
    cost = (1 / len(y)) * ((np.matmul(((-y).T), (np.log(h_theta_x + 1e-5)))) - (
        np.matmul(((1 - y).T), (np.log(1 - h_theta_x + 1e-5)))))
    return cost


def calculate_params_gd(X, y, params, rate):
    for i in range(1200):
        params = params - (rate / len(y)) * (np.matmul(X.T, (calculate_sigmoid(np.matmul(X, params)) - y)))
    return params


def predict_y(X, params):
    mul = np.matmul(X, params)
    res = np.round(calculate_sigmoid(mul))
    return res


def calculate_sigmoid(num):
    # map predicted values to probability
    res = 1 / (1 + np.exp(-num))
    return res

def test(X, y, params):
    new_y = predict_y(X, params)
    score = float(np.sum(new_y == y)) / float(len(y))
    if score > 0.50:
        print("HypsiboasCinerascens")
    else:
        print("HylaMinuta")
def main():
    # D:\UniversityAssignments\IntroToML\hw1_release\Frogs-subsample.csv
    # D:\UniversityAssignments\IntroToML\hw1_release\Frogs.csv
    # read file
    file = input('Enter file path:\n')
    frog = pd.read_csv(file)

    # read features
    X = np.array(frog[['MFCCs_10', 'MFCCs_17']])

    # change target variable to 0 and 1
    z = frog[['Species']]
    y = np.zeros((len(z), 1))
    color = []
    for i in range(len(y)):
        if z['Species'].iloc[i] == "HylaMinuta":
            y[i] = 0
            color.append('r')
        else:
            y[i] = 1
            color.append('b')

    X = np.hstack((np.ones((len(y), 1)), X))
    n = np.size(X, 1)
    params = np.zeros((n, 1))

    # calculate theta/params with learning rate as 2
    params_gd = calculate_params_gd(X, y, params, 2)

    print("Parameters using gradient descent are: \n", params_gd, "\n")

    # calculate accuracy
    new_y = predict_y(X, params_gd)
    score = float(np.sum(new_y == y)) / float(len(y))
    print(f'accuracy is: {score * 100}%')

    # scatter plot
    frog_c1 = frog.iloc[np.where(frog['Species'] == 'HylaMinuta')]
    frog_c2 = frog.iloc[np.where(frog['Species'] == 'HypsiboasCinerascens')]
    x1_f = frog_c1['MFCCs_10']
    y1_f = frog_c1['MFCCs_17']
    x2_f = frog_c2['MFCCs_10']
    y2_f = frog_c2['MFCCs_17']
    plt.scatter(x=x1_f, y=y1_f,s=10,color='red')
    plt.scatter(x=x2_f, y=y2_f,s=10, color='blue')
    plt.xlabel("MFCCs_10")
    plt.ylabel("MFCCs_17")

    # draw decision boundary
    slope = -(params_gd[1] / params_gd[2])
    intercept = -(params_gd[0] / params_gd[2])
    axes = plt.gca()
    axes.autoscale(False)
    x_val = np.array(axes.get_xlim())
    y_val = intercept + (slope * x_val)
    plt.plot(x_val, y_val, c="k")
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.4, 0.5)
    plt.show()



if __name__ == "__main__":
    main()
