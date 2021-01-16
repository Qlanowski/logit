#%%
import matplotlib.pyplot as plt
import matplotlib.colors as cma
import numpy as np
def plot_decision_boundary(X, y, model):
    cMap = cma.ListedColormap(["#e3a77d", "#adb1e0"])
    cMapa = cma.ListedColormap(["#d4732f", "#6b76e8"])

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
    return plt

def plot_iter_cost(model):
    plt.axis('on')
    plt.plot(model.iterationsCosts)
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Wartość funkcji kosztu")
    plt.title(model.method)
    plt.show()

def plot_iter_cost_multiple(results):
    plt.style.use('ggplot')
    plt.axis('on')
    for key, value in results.items():
        plt.plot(value)
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Wartość funkcji kosztu")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Koszt/Liczba iteracji")
    plt.legend(results.keys())
    plt.show()
    return plt

def plot_multiple(results, xlabel, ylabel, title):
    plt.axis('on')
    for key, value in results.items():
        plt.plot(value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(results.keys(),loc='upper right')
    plt.show()

def plot_multiple_tuples(results, xlabel, ylabel, title):
    plt.style.use('ggplot')
    plt.axis('on')
    for key, value in results.items():
        plt.plot(*zip(*value))
    plt.xlabel(xlabel)
    plt.xscale("log")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(results.keys(),loc='upper right')
    plt.xlim(xmin=1)
    plt.show()
    return plt
