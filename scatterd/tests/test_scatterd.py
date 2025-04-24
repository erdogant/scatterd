import unittest
from scatterd import scatterd
import numpy as np
from sklearn import datasets
import matplotlib as mpl

class Testscatterd(unittest.TestCase):

    def test_scatterd(self):
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        labels = iris.target
        custom_cmap = mpl.colors.ListedColormap(['green', 'black', 'blue'])
        
        s = (labels+1) * 200
        random_integers = np.random.randint(0, len(s), size=X.shape[0])
        alpha = np.random.rand(1, X.shape[0])[0][random_integers]
        
        fig, ax = scatterd(X[:,0], X[:,1], labels=labels, marker=labels, gradient='#ffffff', edgecolor='#ffffff', s=s, density=True, alpha=1, cmap=custom_cmap, density_on_top=False, visible=True)
        assert ax is not None
        assert fig is not None
