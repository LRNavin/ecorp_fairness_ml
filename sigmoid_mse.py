# Logistic Regression - Classes 0, 1

import numpy as np
import pandas as pd
import math

from scipy.optimize import fmin_tnc, fmin, fmin_powell, fsolve, minimize, minimize_scalar, fmin_cobyla, fmin_slsqp
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from MLE_Calculator import MLE_Calculator

class LogisticModel:
    """
    Logistic model:
    """

    def __init__(self, X, Y, fair_column):

        self.X = X
        self.Y = Y
        self.theta = np.zeros(self.X.shape[1]) #np.random.random_sample(train_dataset.shape[1]) #np.zeros(train_dataset.shape[1])
        self.loss = 10000000000

        self.fair_column = fair_column

    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        sigm = 1 / (1 + np.exp(-x))
        return sigm

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        prod = np.dot(x, theta) #+ self.bias
        return prod

    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        pred = self.sigmoid(self.net_input(theta, x))
        return pred

    def hard_class_predict(self, soft_pred, labels=(1,2)):
        y_pred = [labels[0] if i <= 0.5 else labels[1] for i in soft_pred]
        return y_pred

    def constraint_1(self, theta, x, y):
        self.theta = theta
        # Implementation of constraint -1
        y_pred = self.hard_class_predict(self.probability(theta, x))
        const_val = MLE_Calculator(y + 1, y_pred, self.fair_column).constraint_1_fair1()
        return const_val

    def constraint_2(self, theta, x, y):
        self.theta = theta
        # Implementation of constraint -1
        y_pred = self.hard_class_predict(self.probability(theta, x))
        const_val = MLE_Calculator(y + 1, y_pred, self.fair_column).constraint_2_fair2()
        return const_val

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        self.theta = theta
        y_pred = self.hard_class_predict(self.probability(theta, x))
        loss = MLE_Calculator(y + 1, y_pred, self.fair_column).calculate_loss()
        print ("Current Loss - " + str(loss))
        self.loss = loss
        return loss

    def gradient_function(self, theta, x, y):
        # Computes the Gradient for all the training samples
        self.theta = new_theta = theta
        y_pred = self.hard_class_predict(self.probability(theta, x))
        grad = MLE_Calculator(y + 1, y_pred, self.fair_column).calculate_grad()

        new_theta -= grad * 0.01
        return new_theta

    def fit(self):
        x, y, theta = self.X, self.Y, self.theta
        # opt_weights = fmin_powell(func=self.cost_function, x0=theta, args=(x, y))

        # Constraint Def:
        constraint_1_dict = {"type": "eq", "fun": self.constraint_1, "args": (x,y)}
        constraint_2_dict = {"type": "eq", "fun": self.constraint_2, "args": (x,y)}

        opt_weights = minimize(fun=self.cost_function, x0=theta, args=(x,y),
                               method="SLSQP",
                               constraints=[constraint_2_dict,constraint_1_dict])

        # opt_weights = fmin_slsqp(func=self.cost_function, x0=theta, args=(x,y), ieqcons=[self.constraint_1, self.constraint_2])

        return opt_weights

    def predict(self, x, parameters):
        theta = parameters[:, np.newaxis]
        return self.probability(theta, x)

# Add Intercept to Log Model via Dataset
fit_intercept = True
def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((X,intercept), axis=1)

#Train Prep - Label & Features split
train_dataset = pd.read_csv("dataset/train.csv", header=None).values
train_label = (train_dataset[:,-1]-1)
train_dataset = train_dataset[:,:-1]

if fit_intercept:
    train_dataset = add_intercept(train_dataset)

# # Normalize numerical cols
# train_dataset[:,:6] = StandardScaler().fit_transform(train_dataset[:,:6])

# Train-Test Split
train_dataset, test_dataset, train_label, test_label = train_test_split(train_dataset, train_label, test_size=0.3, random_state=42)


fair_column = train_dataset[:,12].reshape(train_dataset.shape[0],1)

# train_dataset = np.delete(train_dataset, 12, 1)
# test_dataset  = np.delete(test_dataset, 12, 1)

log_model = LogisticModel(X=train_dataset[:,:-1], Y=train_label, fair_column=fair_column)
optimal_params = log_model.fit().x

final_labels = log_model.hard_class_predict(log_model.predict(x=test_dataset[:,:-1], parameters=optimal_params), labels=(0,1))
print ("Accuracy of Model - " + str(accuracy_score(test_label, final_labels)))

print ("------------------Confusion Matrix------------------")
print (confusion_matrix(test_label, final_labels))

print ("----------------- Model Scores ------------------")
score = precision_recall_fscore_support(test_label, final_labels, average='macro')
print ("Precision - " + str(score[0]))
print ("Recall - " + str(score[1]))
print ("F1 Score - " + str(score[2]))

print ("------------******* Fairness Constraint ********---------------")
hard_labels = log_model.hard_class_predict(log_model.predict(x=test_dataset[:,:-1], parameters=optimal_params))
print ("Constraint 1 : " + str(MLE_Calculator(test_label + 1, hard_labels , fair_column).constraint_1_fair1()))
print ("Constraint 2 : " + str(MLE_Calculator(test_label + 1, hard_labels , fair_column).constraint_2_fair2()))

print ("Weights Shape : " + str(optimal_params.shape))
