# Class for Maximum Likelihood Estimation and Loss Function helper

class MLE_Calculator:

    def __init__(self, classifier_prediction, truth_labels, feature13):

        self.classifier_prediction = classifier_prediction
        self.truth_labels = truth_labels
        self.feature13 = feature13

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

    # Probability that classifier(X) predicts 2 while actual label is 1 and Feature-13 X_13 = 1
    def calculate_p221(self):
        n_p221 = 0
        for index, c_x in enumerate(self.classifier_prediction):
            if c_x == 2 and self.truth_labels[index] == 2 and self.feature13[index] == 1:
                n_p221 = n_p221 + 1

        p221 = n_p221 / len(self.classifier_prediction)
        return p221

    # Probability that classifier(X) predicts 2 while actual label is 1 and Feature-13 X_13 = 1
    def calculate_p222(self):
        n_p222 = 0
        for index, c_x in enumerate(self.classifier_prediction):
            if c_x == 2 and self.truth_labels[index] == 1 and self.feature13[index] == 1:
                n_p222 = n_p222 + 1

        p222 = n_p222 / len(self.classifier_prediction)
        return p222

    # Calculate loss of the current prediction - 3*max(P_2|11, P_2|12) + max(P_2|21, P_2|22)
    def calculate_loss(self, p211, p212, p221, p222):
        loss = (3 * max(p211, p212)) + max(p221, p222)
        return loss
