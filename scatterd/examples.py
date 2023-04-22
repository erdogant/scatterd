from scatterd import scatterd, import_example
import numpy as np

# %%

df = import_example()
fig, ax = scatterd(df['tsneX'],
                   df['tsneY'],
                   labels=df['labx'],
                   density=True,
                   density_on_top=True,
                   args_density={'alpha': 0.3},
                   gradient='#FFFFFF',
                   edgecolor='#FFFFFF',
                   grid=True,
                   fontweight='bold',
                   fontsize=26,
                   )

# %%
# Import example iris dataet
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
labels = iris.target

# Load library
from scatterd import scatterd

# Scatter the results
fig, ax = scatterd(X[:,0], X[:,1], s=250, grid=True)


# %%import some data to play with
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
labels = iris.target

import colourmap
c=colourmap.fromlist(labels)[0]
c[0]=[0,0,0]
c[1]=[0,0,0]
s = (labels+1) * 200
random_integers = np.random.randint(0, len(s), size=X.shape[0])
alpha = np.random.rand(1, X.shape[0])[0][random_integers]

# %%
from scatterd import scatterd
import matplotlib as mpl
custom_cmap = mpl.colors.ListedColormap(['green', 'black', 'blue'])

s = (labels+1) * 200
random_integers = np.random.randint(0, len(s), size=X.shape[0])
alpha = np.random.rand(1, X.shape[0])[0][random_integers]

fig, ax = scatterd(X[:,0], X[:,1], labels=labels, marker=labels, gradient='#ffffff', edgecolor='#ffffff', s=s, density=True, alpha=1, cmap=custom_cmap, density_on_top=False)

# %%
fig, ax = scatterd(X[:,0], X[:,1], labels=None, s=alpha*1000, alpha=alpha)

# %%
# s=np.random.randint(10, 500,len(labels))
fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker=labels.astype(str), s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0])
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, marker=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0])

fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker=np.repeat('X', X.shape[0]))
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, marker=labels, s=300, grid=True)
fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker=labels, s=300)
fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker='s', visible=True)


marker=labels.astype(str)
marker[labels==0]='s'
marker[labels==1]='o'
marker[labels==2]='d'

fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker=marker, visible=True)

# %% Density 3d
fig, ax = scatterd(X[:,0], X[:,1], z=X[:,1], c=[0,0,0], labels=labels, density=True)

# %% Scatter

# Dots foreground
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], labels=None, verbose=4, visible=True,density=True)

# Dots background
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], labels=None, verbose=4, visible=False)
scatterd(X[:,0], X[:,1], c=None, density=True, ax=ax)
fig.set_visible(True)


# %% Scatter

fig, ax = scatterd(X[:,0], X[:,1], c=[1,0,0], labels=None, verbose=4)

fig, ax = scatterd(X[:,0], X[:,1], labels=None, c=[0,0,0], density=True)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, c=[0,0,0], density=True, fontsize=28, legend=False, fontcolor='#000000')
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, c=[0,0,0], density=True, fontsize=28, legend=True, fontcolor='#000000')

fig, ax = scatterd(X[:,0], X[:,1], labels=None, c=None, density=True)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, density=True)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, c=None, density=True)

fig, ax = scatterd(X[:,0], X[:,1], labels=labels, verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=[1,0,0], labels=None, verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], labels=labels, verbose=4, gradient='#ffffff')
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], labels=labels, verbose=4, gradient='#ffffff')
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, verbose=4, gradient='#ffffff')

# Grid options
fig, ax = scatterd(X[:,0], X[:,1], c=c, grid=False, verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=c, grid=True, verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=c, grid='#000000', verbose=4)

# Change figure size
fig, ax = scatterd(X[:,0], X[:,1], verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], labels=labels, verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], X[:,1], c=c, verbose=4)
fig, ax = scatterd(X[:,0], X[:,1], c=[0,0,0], density=True)

fig, ax = scatterd(X[:,0], X[:,1], labels=labels)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, fontcolor='#000000')
fig, ax = scatterd(X[:,0], X[:,1], labels=None, c=[0,0,0])
fig, ax = scatterd(X[:,0], X[:,1], fontcolor=[1,0,0])
fig, ax = scatterd(X[:,0], X[:,1], fontcolor='r')

fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=150, fontcolor='r')
fig, ax = scatterd(X[:,0], X[:,1], c=c, s=150)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, gradient='#FFFFFF', s=150)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, cmap='Set2', s=150)


# Color based on labels
fig, ax = scatterd(X[:,0], X[:,1], s=100, density=True)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, density=True)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, density=True, fontcolor=[0,0,0])

# Set labels
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=100, norm=True, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title')
# Change sizes
s=np.random.randint(10, 500,len(labels))
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True)

fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0])
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, marker=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0])
fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0])

args_density = {'fill': True, 'thresh': 0, 'levels': 100, 'cmap':"mako", 'cbar': False}
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=s, cmap='tab20c', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0], grid=None, args_density=args_density)
fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=0, cmap='tab20c', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0], grid=None, args_density=args_density)

# %%