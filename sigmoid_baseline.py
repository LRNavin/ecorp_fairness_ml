# Logistic Regression - Classes 0, 1
from __future__ import division
import numpy as np
import pandas as pd
import math
from scipy.special import xlogy

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
from sklearn.preprocessing import OneHotEncoder

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

    def logit(self, x):
        # Activation function used to map any real value between 0 and 1
        logit_val = np.log(x/(1-x))
        return self.sigmoid(logit_val)

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        prod = np.dot(x, theta) #+ self.bias
        return prod

    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        pred = self.sigmoid(self.net_input(theta, x))
        return pred

    # Cost Function of basic Logistic Regression
    def log_cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        proba = self.probability(theta, x)
        total_cost = -(1 / m) * np.sum(y * xlogy(np.sign(proba), proba) + (1 - y) * xlogy(np.sign(1 - proba), 1-proba))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def hard_class_predict(self, soft_pred, labels=(1,2)):
        y_pred = [labels[0] if i <= 0.5 else labels[1] for i in soft_pred]
        return y_pred

    def constraint_1(self, theta, x, y):
        # Implementation of constraint -1
        y_pred = self.hard_class_predict(self.probability(theta, x))
        const_val = MLE_Calculator(y + 1, y_pred, self.fair_column).constraint_1_fair1()
        return const_val

    def constraint_2(self, theta, x, y):
        # Implementation of constraint -1
        y_pred = self.hard_class_predict(self.probability(theta, x))
        const_val = MLE_Calculator(y + 1, y_pred, self.fair_column).constraint_2_fair2()
        return const_val

    def cost_function(self, theta, x, y):

        # Computes the cost function for all the training samples
        self.theta = theta
        # For Our Loss fn..,
        y_pred = self.hard_class_predict(self.probability(theta, x))
        loss, taget_loss = MLE_Calculator(y + 1, y_pred, self.fair_column).calculate_loss()
        self.loss = loss

        print ("Target Loss - " + str(taget_loss) + " and Current Loss - " + str(loss))

        return loss
    #
    # def gradient_function(self, theta, x, y):
    #     # Computes the Gradient for all the training samples
    #     self.theta = new_theta = theta
    #     y_pred = self.hard_class_predict(self.probability(theta, x))
    #     grad = MLE_Calculator(y + 1, y_pred, self.fair_column).calculate_grad()
    #
    #     new_theta -= grad * 0.01
    #     return new_theta

    def fit(self):
        x, y, theta = self.X, self.Y, self.theta
        # opt_weights = fmin_powell(func=self.cost_function, x0=theta, args=(x, y))

        # Constraint Def:
        constraint_1_dict = {"type": "eq", "fun": self.constraint_1, "args": (x,y)}
        constraint_2_dict = {"type": "eq", "fun": self.constraint_2, "args": (x,y)}

        # opt_weights = minimize(fun=self.log_cost_function, x0=theta, args=(x,y),
        #                        method="SLSQP",
        #                        constraints=[constraint_2_dict,constraint_1_dict])

        opt_weights = fmin_tnc(func=self.log_cost_function, x0=theta,
                               fprime=self.gradient, args=(x, y.flatten()))[0]

        # opt_weights = fmin_slsqp(func=self.cost_function, x0=theta, args=(x,y), ieqcons=[self.constraint_1, self.constraint_2])

        return opt_weights

    def predict(self, x, parameters):
        theta = parameters[:, np.newaxis]
        return self.probability(theta, x)

# Add Intercept to Log Model via Dataset
fit_intercept = True
select_feature= True
encode_feature= True

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((X,intercept), axis=1)

#Train Prep - Label & Features split
train_dataset = pd.read_csv("dataset/train.csv", header=None).values
train_label = (train_dataset[:,-1]-1)
train_dataset = train_dataset[:,:-1]


#Fairness Required Column  - X_13 - BUFFER
fair_column = train_dataset[:,12].reshape(train_dataset.shape[0],1)

# train_dataset = np.delete(train_dataset, 12, 1)
# test_dataset  = np.delete(test_dataset, 12, 1)

if encode_feature:
    # ************************************ Feature Engineer ************************************
    # Normalize numerical cols
    cont_columns = StandardScaler().fit_transform(train_dataset[:,:6])
    # # One Hot Encode categorical cols
    cat_columns  = OneHotEncoder(handle_unknown='ignore').fit_transform(train_dataset[:,6:]).toarray()
    train_dataset = np.concatenate((cont_columns, cat_columns), axis=1)

if select_feature:
    # ************************************ Feature Selection ************************************
    clf = ExtraTreesClassifier(n_estimators=10)
    clf = clf.fit(train_dataset, train_label)
    train_dataset = SelectFromModel(clf, prefit=True).transform(train_dataset)
    print ("Post Feature Selection, Dataset size - " + str(train_dataset.shape))

if fit_intercept:
    # Add Bias Term/Intercept to Sigmoid function
    train_dataset = add_intercept(train_dataset)

# ADD FairColumn Temprarily - For Dataset Split
train_dataset = np.concatenate((train_dataset, fair_column), axis=1)

# Train-Test Split
train_dataset, test_dataset, train_label, test_label = train_test_split(train_dataset, train_label, test_size=0.3)

#Final Fair Column - For Train (Dirty Work - Remove Temp Fair Column)
fair_column_train   = train_dataset[:,-1]
fair_column_test    = test_dataset[:,-1]
train_dataset = train_dataset[:,:-1]
test_dataset  = test_dataset[:,:-1]

# -1 in all datasets to remove the column of ->
log_model = LogisticModel(X=train_dataset, Y=train_label, fair_column=fair_column_train)
optimal_params = log_model.fit()

print ("------------------Confusion Matrix------------------")
final_labels = log_model.hard_class_predict(log_model.predict(x=test_dataset, parameters=optimal_params), labels=(0,1))
score = precision_recall_fscore_support(test_label, final_labels, average='macro')
print (confusion_matrix(test_label, final_labels))
print ("Precision - " + str(score[0]))
print ("Recall - " + str(score[1]))
print ("F1 Score - " + str(score[2]))

print ("------------******* Fairness Constraint ********---------------")
hard_labels = log_model.hard_class_predict(log_model.predict(x=test_dataset, parameters=optimal_params))
print ("Constraint 1 : " + str(MLE_Calculator(test_label + 1, hard_labels , fair_column_test).constraint_1_fair1()))
print ("Constraint 2 : " + str(MLE_Calculator(test_label + 1, hard_labels , fair_column_test).constraint_2_fair2()))

print ("------------------ Performance Measures ------------------")
loss, target_loss = MLE_Calculator(test_label + 1, hard_labels, fair_column_test).calculate_loss()
print ("Accuracy of Model - " + str(accuracy_score(test_label, final_labels)))
print ("Assignment Loss Value - " + str(target_loss))

print ("Weights Shape : " + str(optimal_params.shape))
