# %%
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from visualize2d import plot_decision_boundary, plot_iter_cost, plot_iter_cost_multiple
from scipy.optimize import minimize
import matplotlib.pyplot as plt

RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, max_iter=10000, regularization='l2', C = 1, tolerance = 1e-9, method="SGD", options=None):
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
        self.theta = np.random.rand(X.shape[1])
        self.currentX = X
        self.currentY = y

        m = X.shape[1]

        if self.method == "SGD":
            self.theta = self.SGD(X, y)
        else:
            res = minimize(fun=self.cost_function, x0=self.theta, args=(X, y), method=self.method, jac=self.gradient, options=self.options, callback=self.callback)
            self.theta = res.x
            #print(res.x)
            #print(res.message)
        
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
def get_dataset(n_samples,n_features,SEED):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1,class_sep=0.7,random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    return X_train, X_test, y_train, y_test
# %%
from sklearn.datasets import make_classification

datasets = [(1000, 2), (1000, 500), (2000, 500)]
methods  = [ "SGD", "CG", "L-BFGS-B"]#"nelder-mead"

for i, d in enumerate(datasets):
    X_train, X_test, y_train, y_test = get_dataset(d[0], d[1], RANDOM_STATE)
    results = {}
    for j, m in enumerate(methods):
        clf = LogisticRegression(method=m).fit(X_train, y_train)
        print(f"Method: {m}, Rows: {d[0]}, Features: {d[1]}, Accuracy: " + str(accuracy_score(y_test, clf.predict(X_test))))
        if(i == 0 and j==0):
            plot_decision_boundary(X_test, y_test, clf)
        results[clf.method] = clf.iterationsCosts

    iter_cost_plt = plot_iter_cost_multiple(results)
    iter_cost_plt.savefig(f'charts/iter_cost_rows_{d[0]}_features_{d[1]}.svg')

# %%
#plot_iter_cost(clf_our)
