# %%
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from visualize2d import plot_decision_boundary, plot_iter_cost, plot_iter_cost_multiple
from scipy.optimize import minimize
import matplotlib.pyplot as plt

RANDOM_STATE = 0
METHOD = "CG"

class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, max_iter=1200, regularization='l2', C = 1, tolerance = 1e-4, method="SGD", options=None):
        self.learning_rate  = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.C = C
        self.lam = 1/C
        self.tolerance = tolerance
        self.method = method
        self.options = options
        self.currentX = None
        self.currentY = None
        self.iterationsCosts = []

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        self.currentX = X
        self.currentY = y

        m = X.shape[1]

        if self.method == "SGD":
            self.theta = self.SGD(X, y)
        else:
            res = minimize(fun=self.cost_function, x0=self.theta, args=(X, y), method=self.method, jac=self.gradient, options=self.options, callback=self.callback)
            self.theta = res.x
            print(res.x)
            print(res.message)
        
        return self

    def callback(self, xk):
        cost = self.cost_function(xk, self.currentX, self.currentY)
        self.iterationsCosts.append(cost)

    def SGD(self, X, y):
        theta = self.theta
        for _ in range(self.max_iter):
            step = self.learning_rate * self.gradient(theta, X, y)
            if np.all(abs(step) >= self.tolerance):
                theta -= step
            else:
                break
            self.callback(theta)

        return theta

    def cost_function(self, theta, X, y):
        m = X.shape[1]
        hx = self.h(X @ theta)

        res = 1 / m * np.sum(-y * np.log(hx) - (1-y) * np.log(1-hx)) + self.lam/(2*m)*np.sum(theta**2)
        return res

    def gradient(self, theta, X, y):
        m = X.shape[1]
        y_hat = self.h(X @ theta)
        errors = y_hat - y

        res = 1/m* ((X.T @ errors) + self.lam * np.sum(theta))
        return res

    def predict_proba(self, X):
        return self.h(X @ self.theta)

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def h(self, z):
        return 1 / (1 + np.exp(-z))

# %% 
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1,random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf_our = LogisticRegression(method=METHOD).fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf_our.predict(X_test))))
plot_decision_boundary(X_test, y_test, clf_our)


# %%
#plot_iter_cost(clf_our)

# %%
results = {}

clf = LogisticRegression(method="SGD").fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
results[clf.method] = clf.iterationsCosts

clf = LogisticRegression(method="CG").fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
results[clf.method] = clf.iterationsCosts

clf = LogisticRegression(method="L-BFGS-B").fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
results[clf.method] = clf.iterationsCosts

clf = LogisticRegression(method="SLSQP").fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
results[clf.method] = clf.iterationsCosts

plot_iter_cost_multiple(results)