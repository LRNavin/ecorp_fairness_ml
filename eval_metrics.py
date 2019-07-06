# Evaluate Metrics from CV

from __future__ import division
import numpy as np
import pandas as pd
import math
import csv
import os
import errno
from os import listdir
from os.path import isfile, join

find_mean_std = True


folder = "metrics/best_test_model/"
metrics_dict = {}


for filename in listdir(folder):
    if isfile(join(folder, filename)):
        path = join(folder, filename)
        # print("File Found - " + str(filename))
        metric_list = pd.read_csv(path, header=None).values

        avg = np.mean(metric_list)
        std = np.std(metric_list)
        metrics_dict[filename] = {"mean":avg, "variance":std}
        print ("Metrics of " + str(filename) + ": Mean - " + str(avg) + ", Variance - " + str(std))

print(metrics_dict)