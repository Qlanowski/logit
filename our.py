# %%
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from visualize2d import plot_decision_boundary

class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, max_iter=1200, regularization='l2', C = 1, tolerance = 1e-4):
        self.learning_rate  = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.C = C
        self.tolerance = tolerance
    
    def fit(self, X, y):
      
        self.theta = np.zeros(X.shape[1])
        m = X.shape[1]
        reg = self.C / m
        for _ in range(self.max_iter):
            y_hat = self.h(X @ self.theta)
            errors = y - y_hat

            step = self.learning_rate * (reg * (X.T @ errors) + np.sum(self.theta))

            if np.all(abs(step) >= self.tolerance):
                self.theta -= step
            else:
                break
                
        return self

    def predict_proba(self, X):
        return self.h(X @ self.theta)
    
    def predict(self, X):
        return np.round(self.predict_proba(X))
        
    def h(self, z):
        return 1 / (1 + np.exp(z))


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf_our = LogisticRegression().fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf_our.predict(X_test))))
plot_decision_boundary(X_test, y_test, clf_our)


# %%

# %%
