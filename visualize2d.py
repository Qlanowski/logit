#%%
import matplotlib.pyplot as plt
import matplotlib.colors as cma
import numpy as np
def plot_decision_boundary(X, y, model,path=None):
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
    plt.tight_layout()
    if path != None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()


def plot_iter_cost_multiple(results, xlabel, ylabel, title, path=None):
    plt.style.use('ggplot')
    plt.axis('on')
    for key, value in results.items():
        plt.plot(value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(results.keys())
    plt.tight_layout()
    if path != None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

def plot_multiple_tuples(results, xlabel, ylabel, title, path=None):
    plt.style.use('ggplot')
    plt.axis('on')
    for key, value in results.items():
        time = [v[1]*1000 for v in value]
        plt.plot(time)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(results.keys(),loc='upper right')
    plt.xlim(xmin=1)
    plt.tight_layout()
    if path != None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

def plot_time_cost_tuples(times, costs, xlabel, ylabel, title, path=None):
    plt.style.use('ggplot')
    plt.axis('on')
    for key, value in times.items():
        cost = costs[key]
        time = [v[1]*1000 for v in value]
        plt.plot(time, cost)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(times.keys(),loc='upper right')
    plt.xlim(xmin=1)
    plt.tight_layout()
    if path != None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()