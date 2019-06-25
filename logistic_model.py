# Logistic Regression - Classes 0, 1

import numpy as np
import pandas as pd
import math

from sklearn.metrics import mean_squared_error

from scipy.optimize import fmin_tnc, fmin, fmin_powell, fsolve, minimize, minimize_scalar

from MLE_Calculator import MLE_Calculator

class LogisticModel:
    """
    Logistic model:
    """
    def __init__(self, X, Y):

        self.X = X
        self.Y = Y
        self.theta = np.random.randint(10, size=self.X.shape[1])
        self.loss = 0

    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        sigm = 1 / (1 + np.exp(-x))
        return sigm

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        prod = np.dot(x, theta)
        return prod

    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        pred = self.sigmoid(self.net_input(theta, x))
        return pred

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        # print ("Diff Theta - " + str(max(self.theta - theta)))
        self.theta = theta
        y_pred = np.log(self.probability(theta, x))
        loss = MLE_Calculator(y + 1, y_pred + 1, x[:,12]).calculate_loss
        print ("Current Loss - " + str(loss))
        # print ("Diff Loss - " + str(loss - self.loss))
        self.loss = loss
        return loss

    def fit(self):
        x, y, theta = self.X, self.Y, self.theta
        opt_weights = fsolve(func=self.cost_function, x0=theta, args=(x, y.flatten()))
        return opt_weights

    def predict(self, x, parameters):
        theta = parameters[:, np.newaxis]
        return self.probability(theta, x)

#Train Prep - Label & Features split
train_dataset = pd.read_csv("dataset/train.csv", header=None).values
train_label = train_dataset[:,-1]-1
train_dataset = train_dataset[:,:-1]

log_model = LogisticModel(X=train_dataset, Y=train_label)
optimal_params = log_model.fit()
print(optimal_params.shape)
prediction = log_model.predict(x=train_dataset, parameters=optimal_params)
print (prediction)