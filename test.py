# %% Tests
from scatterd.scatterd import scatterd

# %%import some data to play with
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# %%
# Simple scatter
scatterd(X[:,0], X[:,1])
# Color based on labels
scatterd(X[:,0], X[:,1], label=y, s=100)
# Set labels
scatterd(X[:,0], X[:,1], label=y, s=100, norm=True, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title')
# Change sizes
s=np.random.randint(10,200,len(y))
scatterd(X[:,0], X[:,1], label=y, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, figsize=(15,10))
# Change figure size
scatterd(X[:,0], X[:,1], figsize=(25,15))

# %%
