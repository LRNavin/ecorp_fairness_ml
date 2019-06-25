# Class for Maximum Likelihood Estimation and Loss Function helper
from __future__ import division
import math

class MLE_Calculator:

    def __init__(self, truth_labels, classifier_prediction, feature13):

        self.classifier_prediction  = classifier_prediction
        self.truth_labels           = truth_labels
        self.feature13              = feature13

        self.dataset                = feature13

        # Return Loss kind Flag
        self.mse_loss = True

    # Probability that classifier(X) predicts 2 while actual label is 1 and Feature-13 X_13 = 1
    def calculate_p211(self):
        n_p211 = 0
        for index, c_x in enumerate(self.classifier_prediction):
            if c_x == 2 and self.truth_labels[index] == 1 and self.feature13[index] == 1:
                n_p211 = n_p211 + 1

        p211 = n_p211/len(self.classifier_prediction)
        return p211

    # Probability that classifier(X) predicts 2 while actual label is 1 and Feature-13 X_13 = 2
    def calculate_p212(self):
        n_p212 = 0
        for index, c_x in enumerate(self.classifier_prediction):
            if c_x == 2 and self.truth_labels[index] == 1 and self.feature13[index] == 2:
                n_p212 = n_p212 + 1

        p212 = n_p212 / len(self.classifier_prediction)
        return p212

    # Probability that classifier(X) predicts 1 while actual label is 2 and Feature-13 X_13 = 1
    def calculate_p121(self):
        n_p121 = 0
        for index, c_x in enumerate(self.classifier_prediction):
            if c_x == 1 and self.truth_labels[index] == 2 and self.feature13[index] == 1:
                n_p121 = n_p121 + 1

        p121 = n_p121 / len(self.classifier_prediction)
        return p121

    # Probability that classifier(X) predicts 1 while actual label is 2 and Feature-13 X_13 = 2
    def calculate_p122(self):
        n_p122 = 0
        for index, c_x in enumerate(self.classifier_prediction):
            if c_x == 1 and self.truth_labels[index] == 2 and self.feature13[index] == 2:
                n_p122 = n_p122 + 1

        p122 = n_p122 / len(self.classifier_prediction)
        return p122

    # Calculate loss of the current prediction - 3*max(P_2|11, P_2|12) + max(P_2|21, P_2|22)
    def calculate_loss(self):
        p211, p212, p121, p122 = self.calculate_p211(),\
                                 self.calculate_p212(),\
                                 self.calculate_p121(),\
                                 self.calculate_p122()

        loss = (3 * max(p211, p212)) + max(p121, p122)
        true_p211, true_p212, true_p121, true_p122 = self.true_loss_value()
        true_loss = (3 * max(true_p211, true_p212)) + max(true_p121, true_p122)

        # print("Sum Test - " + str(true_p121 + true_p212 + true_p122 + true_p211))

        if self.mse_loss:
            # final_loss = (3 * ( 0.5 * math.pow(max(p211, p212) - max(true_p211, true_p212),2)))\
            #                 + ( 0.5 * math.pow(max(p121, p122) - max(true_p121, true_p122),2))

            final_loss = 0.5 * math.pow((true_loss - loss),2)
        else:
            final_loss = loss

        return final_loss

    # Gradient of loss fn. If mse
    def calculate_grad(self):

        p211, p212, p121, p122 = self.calculate_p211(), \
                                 self.calculate_p212(), \
                                 self.calculate_p121(), \
                                 self.calculate_p122()

        loss = (3 * max(p211, p212)) + max(p121, p122)
        true_p211, true_p212, true_p121, true_p122 = self.true_loss_value()
        true_loss = (3 * max(true_p211, true_p212)) + max(true_p121, true_p122)

        gradient = true_loss - loss

        return gradient

    # Implement constraint-1=> p211 = p212 => (p211 - p212 = 0)[Fair on X-13 w.r.t True Class 1 & wrong predict by c(x)]
    def constraint_1_fair1(self):
        p211, p212 = self.calculate_p211(), \
                     self.calculate_p212()
        const_1 = p211 - p212
        return const_1

    # Implement constraint-1=> p121 = p122 => (p121 - p122 = 0)[Fair on X-13 w.r.t True Class 2 & wrong predict by c(x)]
    def constraint_2_fair2(self):
        p121, p122 = self.calculate_p121(), \
                     self.calculate_p122()
        const_2 = p121 - p122
        return const_2


    def true_loss_value(self):

        true_p211 = 0
        true_p212 = 0

        true_p121 = 0
        true_p122 = 0

        for i, feature in enumerate(self.feature13):

            # Right Prediction of p211 as p111 - Truth in Dataset
            if self.truth_labels[i] == 1 and self.truth_labels[i] == 1 and feature == 1:
                true_p211 += 1

            # Right Prediction of p212 as p112 - Truth in Dataset
            if self.truth_labels[i] == 1 and self.truth_labels[i] == 1 and feature == 2:
                true_p212 += 1

            # Right Prediction of p121 as p221 - Truth in Dataset
            if self.truth_labels[i] == 2 and self.truth_labels[i] == 2 and feature == 1:
                true_p121 += 1

            # Right Prediction of p121 as p221 - Truth in Dataset
            if self.truth_labels[i] == 2 and self.truth_labels[i] == 2 and feature == 2:
                true_p122 += 1

        true_p211 = true_p211 / len(self.feature13)
        true_p212 = true_p212 / len(self.feature13)

        true_p121 = true_p121 / len(self.feature13)
        true_p122 = true_p122 / len(self.feature13)

        # loss = (3 * max(true_p211, true_p212)) + max(true_p121, true_p122)

        return true_p211, true_p212, true_p121, true_p122