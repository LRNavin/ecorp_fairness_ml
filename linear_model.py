import numpy as np
import pandas as pd
import math

from MLE_Calculator import MLE_Calculator
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class LinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    """

    def __init__(self, X=None, Y=None, loss_function=None):

        self.loss_function = loss_function
        self.X = X
        self.Y = Y
        self.beta = np.random.random_sample((self.X.shape[1])) #np.array([1] * self.X.shape[1])

        # print (self.beta.shape)


    def model_error(self):
        error = self.loss_function(self.predict(self.X), self.Y)
        return(error)

    def l2_loss(self, beta):
        self.beta = beta
        return(self.model_error() )

    def mle_loss(self, beta):
        self.beta = beta
        prediction = self.predict(self.X)
        return max_likelihood_loss(self.Y, prediction, self.X[:, 12])

    def sigmoid(self, x_val):
        return 1 / (1 + math.exp(-x_val))

    def predict(self, X):
        # print (self.beta)
        # print (self.X[1,:])

        val = np.matmul(X, self.beta)
        norm_val = (val - min(val))/(max(val) - min(val))

        print("Number of lows - " + str(sum(i <= 0.5 for i in norm_val)))

        prediction = [1 if  i <= 0.5 else 2 for i in norm_val]
        print ("Actual 5 label - " + str(self.Y[:10]))
        print ("First 5 prediciton - " + str(prediction[:10]))
        return prediction

    def fit(self, maxiter=1):
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        iter_counter = maxiter
        while (iter_counter > 0):

            #Update Beta -- How ?
            res = minimize(self.l2_loss, self.beta,
                           method='BFGS', options={'maxiter': 500})
            self.beta = res.x
            iter_counter = iter_counter - 1


def mean_absolute_percentage_error(y_true, y_pred, sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)

    if np.any(y_true == 0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true == 0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)

    if type(sample_weights) == type(None):
        loss = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        loss = 100 / sum(sample_weights) * np.dot(
            sample_weights, (np.abs((y_true - y_pred) / y_true)))

    print ("Current Loss - " + str(loss))

    return loss

def max_likelihood_loss(y_true, y_pred, X_13):
    # loss = MLE_Calculator(y_true, y_pred, X_13).calculate_loss()
    loss = mean_squared_error(y_true, y_pred)
    print ("Current Loss - " + str(loss))
    return loss

#Train Prep - Label & Features split
train_dataset = pd.read_csv("dataset/train.csv", header=None).values
train_label = train_dataset[:,-1]
train_dataset = train_dataset[:,:-1]

linear_model = LinearModel(X=train_dataset, Y=train_label, loss_function=mean_absolute_percentage_error)
linear_model.fit()
print linear_model.beta