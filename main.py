# %%
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

class DataLoader(object):
    def __init__(self, path):
        self.df = pd.read_csv(path, quotechar='"', delimiter=",")
        self.features = ['x', 'y']
        self.target = ['cls']

    def generate_test_train_datasets(self, test_size=0.25):
        X_train, X_test, y_train, y_test = train_test_split(self.df[self.features], self.df[self.target], test_size=test_size, random_state=1)
        return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().ravel(), y_test.to_numpy().ravel()

class LogisticRegression(object):
    """
    Logistic Regression Classifier

    Parameters
    ----------
    learning_rate : int or float, default=0.1
        The tuning parameter for the optimization algorithm (here, Gradient Descent) 
        that determines the step size at each iteration while moving toward a minimum 
        of the cost function.

    max_iter : int, default=100
        Maximum number of iterations taken for the optimization algorithm to converge
    
    penalty : None or 'l2', default='l2'.
        Option to perform L2 regularization.

    C : float, default=0.1
        Inverse of regularization strength; must be a positive float. 
        Smaller values specify stronger regularization. 

    tolerance : float, optional, default=1e-4
        Value indicating the weight change between epochs in which
        gradient descent should terminated. 
    """

    def __init__(self, learning_rate=0.1, max_iter=100, regularization='l2', C = 0.1, tolerance = 1e-4):
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.regularization = regularization
        self.C              = C
        self.tolerance      = tolerance
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        """
        self.theta = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        print(X.shape)
        print(y.shape)

        for _ in range(self.max_iter):
            y_hat = self.__sigmoid(X @ self.theta)
            errors = y - y_hat
            N = X.shape[1]

            if self.regularization is not None:
                delta_grad = self.learning_rate * ((self.C * (X.T @ errors)) + np.sum(self.theta))
            else:
                delta_grad = self.learning_rate * (X.T @ errors)

            if np.all(abs(delta_grad) >= self.tolerance):
                self.theta -= delta_grad / N
            else:
                break
                
        return self

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        probs : array-like of shape (n_samples,)
            Returns the probability of each sample.
        """
        return self.__sigmoid((X @ self.theta[1:]) + self.theta[0])
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        labels : array, shape [n_samples]
            Predicted class label per sample.
        """
        return np.round(self.predict_proba(X))
        
    def __sigmoid(self, z):
        """
        The sigmoid function.

        Parameters
        ------------
        z : float
            linear combinations of weights and sample features
            z = w_0 + w_1*x_1 + ... + w_n*x_n
        Returns
        ---------
        Value of logistic function at z
        """
        return 1 / (1 + expit(-z))

    def get_params(self):
        """
        Get method for models coeffients and intercept.

        Returns
        -------
        params : dict
        """
        try:
            params = dict()
            params['intercept'] = self.theta[0]
            params['coef'] = self.theta[1:]
            return params
        except:
            raise Exception('Fit the model first!')
    def accuracy(self, y, y_hat):
        return 1 - accuracy_score(y, y_hat)

import matplotlib.pyplot as plt
import matplotlib.colors as cma
def plot_decision_boundary(X, y, model):
    cMap = cma.ListedColormap(["#6b76e8", "#c775d1"])
    cMapa = cma.ListedColormap(["#c775d1", "#6b76e8"])

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    plt.figure(1, figsize=(8, 6), frameon=True)
    plt.axis('off')
    plt.pcolormesh(xx, yy, Z, cmap=cMap)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker = "o", edgecolors='k', cmap=cMapa)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
#%%
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf_our_random = LogisticRegression().fit(X_train, y_train)
print("Accuracy: " + str(clf_our_random.accuracy(y_test, clf_our_random.predict(X_test))))
#plot_decision_boundary(X_test, y_test, clf_our_random)

from sklearn.linear_model import LogisticRegression as sklr
clf_sk_random = sklr(random_state=0).fit(X_train, y_train)
print("Accuracy: " + str(clf_sk_random.score(X_test,y_test)))
#plot_decision_boundary(X_test, y_test, clf_sk_random)

#%%
dataLoader = DataLoader("./data/classificationData/data.simple.train.1000.csv")
X_train, X_test, y_train, y_test = dataLoader.generate_test_train_datasets()

clf_our_real = LogisticRegression().fit(X_train, y_train)
print("Accuracy: " + str(clf_our_real.accuracy(y_test, clf_our_real.predict(X_test))))

clf_sk_real = sklr(random_state=0).fit(X_train, y_train)
print("Accuracy: " + str(clf_sk_real.score(X_test, y_test)))