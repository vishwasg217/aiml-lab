import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for multi-class classification.

    Attributes:
        classes (numpy.ndarray): Array of unique class labels.
        means (numpy.ndarray): Means of features for each class.
        variances (numpy.ndarray): Variances of features for each class.
        priors (numpy.ndarray): Priors (class probabilities) for each class.
    """

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model to the training data.

        Parameters:
            X (numpy.ndarray): Training data features.
            y (numpy.ndarray): Training data labels.

        Returns:
            None
        """
        n_samples, n_features = X.shape

        # initialize the classes, means, variances, and priors
        self.classes = np.unique(y)
        self.means = np.zeros((len(self.classes), n_features))
        self.variances = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        # calculate the mean, variance, and prior for each class
        for class_idx, c in enumerate(self.classes):
            X_for_c = X[y == c]
            self.means[class_idx] = X_for_c.mean(axis=0)
            self.variances[class_idx] = X_for_c.var(axis=0)
            self.priors[class_idx] = X_for_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict the class labels for input data using trained model.

        Parameters:
            X (numpy.ndarray): Input data features.

        Returns:
            numpy.ndarray: Predicted class labels for each input sample.
        """
        return [self._classify_sample(x) for x in X]
    
    def _classify_sample(self, x):
        """
        Classify a single input sample based on trained model.

        Parameters:
            x (numpy.ndarray): Input sample features.

        Returns:
            int: Predicted class label for the input sample.
        """
        posteriors = []

        # calculate posterior probability for each class
        for class_idx, c in enumerate(self.classes):
            # we use log probabilities to avoid multiplication of small probabilities as it can lead to underflow
            prior = np.log(self.priors[class_idx])
            likelihood = np.sum(np.log(self._calculate_likelihood(x, class_idx)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def _calculate_likelihood(self, x, class_idx):
        """
        Calculate the likelihood of input features given a class.

        Parameters:
            x (numpy.ndarray): Input sample features.
            class_idx (int): Index of the class.

        Returns:
            numpy.ndarray: Likelihoods of the input features given the class.
        """
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# iris = load_iris()

# X = iris.data
# y = iris.target


diabetes = pd.read_csv('diabetes.csv')
X = diabetes.iloc[:, :-1].values
y = diabetes.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = gnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1:", f1_score(y_test, y_pred, average='macro'))