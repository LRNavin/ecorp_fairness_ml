from autograd import grad
import autograd.numpy as grad_np
from sklearn.metrics import accuracy_score
from autograd.numpy.numpy_boxes import ArrayBox
from mpmath import rand
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import math

from MLE_Calculator import MLE_Calculator


def sigmoid(x):
    sigm = 0.5 * (grad_np.tanh(x / 2.) + 1)
    return sigm

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    proba = sigmoid(grad_np.dot(inputs, weights))
    return proba

# def training_loss(weights):
#     # Training loss is the negative log-likelihood of the training labels.
#     #
#     if type(weights) == ArrayBox:
#         weights = weights._value
#     #
#     # preds = logistic_predictions(weights, train_dataset)
#     # label_probabilities = preds * train_label + (1 - preds) * (1 - train_label)
#     #
#     # loss = grad_np.log(label_probabilities)
#     # # loss = [0 if i == -np.inf else i for i in loss]
#     # loss = -grad_np.sum(loss)
#     #
#     # # loss = mean_squared_error(train_label, preds)#MLE_Calculator(train_label + 1, preds + 1, train_dataset[:,12]).calculate_loss
#     # print ("Current Loss - " + str(loss))
#     #
#     # return loss
#     # Training loss is the negative log-likelihood of the training labels.
#     # preds = logistic_predictions(weights, train_dataset)
#     # label_probabilities = preds * train_label + (1 - preds) * (1 - train_label)
#     # label_probabilities = [np.log(i) if i != 0 else i for i in label_probabilities]
#     # loss  = -np.sum(label_probabilities)
#     # print ("Current Loss - " + str(loss))
#     # return loss
#
#     preds = logistic_predictions(weights, train_dataset)
#     label_probabilities = preds * train_label + (1 - preds) * (1 - train_label)
#     # label_probabilities = [0 if i == 0 else np.log(i) for i in label_probabilities]
#     return -np.sum((label_probabilities))
#

def training_loss(weights):

    # if type(weights) == ArrayBox:
    #     weights = weights._value

    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, train_dataset)

    # label_probabilities = preds * train_label + (1 - preds) * (1 - train_label)
    # label_probabilities = [i if i < 0.000000001 else grad_np.log(i) for i in label_probabilities]
    loss = MLE_Calculator(train_label + 1, preds, train_dataset[:, 12]).calculate_loss()

    loss = grad_np.sum([loss])
    # loss = loss/len(label_probabilities)

    print ("Current Loss : " + str(loss))

    return loss





#Train Prep - Label & Features split
train_dataset = pd.read_csv("dataset/train.csv", header=None).values
train_label = train_dataset[:,-1]-1
train_dataset = train_dataset[:,:-1]

# Normalize numerical cols
train_dataset[:,:6] = StandardScaler().fit_transform(train_dataset[:,:6])

train_dataset, test_dataset, train_label, test_label = train_test_split(train_dataset, train_label, test_size=0.3, random_state=42)

print ("Normalised Dataset")
print (train_dataset.shape)
print (train_dataset[1,:])

# # Build a toy dataset.
# train_dataset = np.array([[0.52, 1.12,  0.77],
#                    [0.88, -1.08, 0.15],
#                    [0.52, 0.06, -1.30],
#                    [0.74, -2.49, 1.39]])
# train_label = np.array([True, True, False, True])

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.random.random_sample(train_dataset.shape[1])#np.zeros(train_dataset.shape[1])+1

print("Initial loss:", training_loss(weights))
for i in range(100):
    print ("Iteration " + str(i) )
    gradient = training_gradient_fun(weights)
    weights -= gradient  * 0.01

print("Trained loss:", training_loss(weights))

preds = logistic_predictions(weights, test_dataset)

final_labels = [0 if i<=0.5 else 1 for i in preds]
print ("Accuracy of Model - " + str(accuracy_score(test_label, final_labels)))



# DUMMIESSSSS


# print("----------------** Feature Importance **------------------")
# print(clf.feature_importances_.shape)
# print(clf.feature_importances_)
# model = SelectFromModel(clf, prefit=True)
# train_dataset = model.transform(train_dataset)
#


# Feature Imporatance Checker ---
# clf = ExtraTreesClassifier(n_estimators=10)
# clf = clf.fit(train_dataset, train_label)
#
# # Feat Select -
# train_dataset = SelectKBest(chi2, k=10).fit_transform(train_dataset, train_label)
