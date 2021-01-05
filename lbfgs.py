
#%%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from visualize2d import plot_decision_boundary

X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=5 ,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.model_selection import train_test_split
clf_lbfgs = LogisticRegression(solver="lbfgs", penalty="l2", C=1, fit_intercept=False, max_iter=100).fit(X_train, y_train)
print("Accuracy: " + str(accuracy_score(y_test, clf_lbfgs.predict(X_test))))
plot_decision_boundary(X_test,y_test,clf_lbfgs)
# %%
