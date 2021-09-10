import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def decision_regions(X, y, classifier, resolution=0.02):

    # markers and color map
    markers = ('s', 'x', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'graph', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot decision regions
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1   

    # generate grid point
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), 
        np.arange(x2_min, x2_max, resolution)
    )

    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # plot contour line of grid points
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)


    # range of axis
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0], 
            y=X[y == cl, 1], 
            alpha=0.8, 
            c=colors[idx], 
            marker=markers[idx], 
            label=cl, 
            edgecolor='black'
            )