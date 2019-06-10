import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


#Train Prep - Label & Features split
train_dataset = pd.read_csv("dataset/train.csv", header=None).values
train_label = train_dataset[:,-1]
train_dataset = train_dataset[:,:-1]

# Test set without 14th Parameter
test_dataset = pd.read_csv("dataset/test.csv", header=None).values

print("Training set size - " + str(len(train_dataset)) + "\n" + "Testing set size - " + str(len(test_dataset)))
print("Shape of Training Set - " + str(train_dataset.shape) + "\n" + "Shape of Testing Set - " + str(test_dataset.shape))


# TSNE View
X_embedded = TSNE(n_components=2).fit_transform(train_dataset)


# Plot
plt.scatter(X_embedded[:,1], X_embedded[:,0], c=train_label)
plt.title('TSNE Train Dataset')
plt.show()