"""Scatterplot."""
# --------------------------------------------------
# Name        : scatterd.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/scatterd
# Licence     : See licences
# --------------------------------------------------

# %% Libraries
import colourmap as colourmap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% Main
def scatterd(x, y, s=50, c=[0, 0, 0], labels=None, density=False, norm=False, cmap='Set1', figsize=(25, 15),
             xlabel='x-axis', ylabel='y-axis', title='', fontsize=24, fontcolor=None, axiscolor='#dddddd',
             args_density = {'cmap': 'Reds', 'shade': True, 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': True, 'fill': True}):
    """Make scaterplot.

    Parameters
    ----------
    x : numpy array
        1D Coordinates.
    y : numpy array
        1D Coordinates.
    s: Int or list/array of sizes with same size as X
        Size of the samples.
    c: list/array of RGB colors with same size as X
        Color of samples in RGB colors.
    labels: list of labels with same size as X
        labels of the samples.
    density : Bool (default: False)
        Include the kernel density in the scatter plot.
    xlabel : String, optional
        Xlabel. The default is None.
    ylabel : String, optional
        Ylabel. The default is None.
    title : String, optional
        Title of the figure. The default is None.
    fontsize : String, optional
        The fontsize of the y labels that are plotted in the graph. The default is 16.
    fontcolor: list/array of RGB colors with same size as X (default : None)
        None : Use same colorscheme as for c
        [0,0,0] : If the input is a single color, all fonts will get this color.
    norm : Bool, optional
        Normalize datapoints. The default is True.
    cmap : String, optional
        'Set1'       (default)
        'Set2'
        'rainbow'
        'bwr'        Blue-white-red
        'binary' or 'binary_r'
        'seismic'    Blue-white-red
        'Blues'      white-to-blue
        'Reds'       white-to-red
        'Pastel1'    Discrete colors
        'Paired'     Discrete colors
        'Set1'       Discrete colors
    figsize : tuple, optional
        Figure size. The default is (15,10).
    args_density : dict()
        Dictionary containing arguments for kernel density plotting.

    References
    -------
    Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

    Returns
    -------
    Fig, ax

    """
    if len(x)!=len(y): raise Exception('[scatterd] >Error: X should have same length as Y.')
    if s is None: raise Exception('[scatterd] >Error: input parameter s(ize) should be not None.')
    if c is None: raise Exception('[scatterd] >Error: input parameter c(olors) should be not None.')
    if not isinstance(s, int) and len(s)!=len(x): raise Exception('[scatterd] >ERROR: input parameter s(ize) should be of same size of X.')

    # Defaults
    defaults_kde = {'cmap': 'Reds', 'shade': True, 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': True, 'fill': True}
    args_density = {**defaults_kde, **args_density}

    # Preprocessing
    X, labels = _preprocessing(x, y, labels, norm)

    # Figure properties
    c_rgb, fontcolor = _set_colors(X, labels, fontcolor, c, cmap)

    # Bootup figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter
    ax.scatter(X[:, 0], X[:, 1], facecolor=c_rgb, s=s, edgecolor='None')

    # Add density to scatterplot
    if density:
        ax = sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, **args_density)

    # Set figure properties
    ax = _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, axiscolor, ax)

    # Show figure
    # fig.show()
    # fig.canvas.draw()

    # Return
    return fig, ax


# %% Setup figure properties
def _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, axiscolor, ax):
    # Set axis fontsizes
    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 16}
    matplotlib.rc('font', **font)

    # Plot labels
    if (labels is not None) and (fontcolor is not None):
        for uilabel in fontcolor.keys():
            XYmean = np.mean(X[labels==uilabel, :], axis=0)
            ax.text(XYmean[0], XYmean[1], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': 'bold', 'size': fontsize})

    # Labels on axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    # set background to none
    ax.set_facecolor('none')
    # Set grid and axis to grey
    if axiscolor is not None:
        ax.grid(True, color=axiscolor)

    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color(axiscolor)

    return ax


# %% Setup colors
def _set_colors(X, labels, fontcolor, c, cmap):
    # Create unqiue colors for labels
    if len(np.unique(labels))>1:
        c_rgb, _ = colourmap.fromlist(labels, cmap=cmap, method='matplotlib')
    else:
        # if len(c)!=3: raise Exception('[scatterd] >Error: input parameter [c] should be in the form of RGB coloring, such as [0, 0, 0]')
        c_rgb = c

    # Set fontcolor
    fontcolor = _fontcolor(fontcolor, labels, X, cmap)
    # Return
    return c_rgb, fontcolor


# %% Fontcolor
def _preprocessing(x, y, labels, norm):
    # Combine into array
    X = np.c_[x, y]
    # Normalize data
    if norm:
        X = _normalize(X)
    # Labels
    if labels is None:
        labels=np.zeros_like(y)
    # Return
    return X, labels


# %% Fontcolor
def _normalize(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    return (X - x_min) / (x_max - x_min)


# %% Fontcolor
def _fontcolor(fontcolor, label, X, cmap, verbose=3):
    # Set font colors
    if (fontcolor is None) and (label is None):
        fontcolor = {np.unique(label)[0]: [0, 0, 0]}
    elif (fontcolor is None) and (len(np.unique(label))==1):
        pass
    elif (fontcolor is not None) and (len(fontcolor)==3 or len(fontcolor)==1):
        fontcolor_dict = {}
        for lab in np.unique(label):
            fontcolor_dict[lab]=fontcolor
        fontcolor=fontcolor_dict
    elif (fontcolor is not None) and (label is None):
        if verbose>=2: print('[scatterd] >Warning : Without label, there is no font(color) to print.')
    elif (fontcolor is None) and (label is not None):
        _, fontcolor = colourmap.fromlist(label, cmap=cmap, method='matplotlib')
    elif (fontcolor is not None) and (label is not None) and (len(fontcolor)==X.shape[0]):
        _, fontcolor = colourmap.fromlist(fontcolor, cmap=cmap, method='matplotlib')
    elif (fontcolor is not None) and (label is not None) and ((isinstance(fontcolor[0], int)) or (isinstance(fontcolor[0], float))):
        _, tmpcolors = colourmap.fromlist(label, cmap=cmap, method='matplotlib')
        # list(map(lambda x: tmpcolors.update({x: fontcolor}), [*tmpcolors.keys()]))
        fontcolor = tmpcolors
    else:
        raise Exception('[scatterd] >ERROR : fontcolor input is not correct.')

    return fontcolor


# %% Density
# def coord2density(X, kernel='gaussian', metric='euclidean', ax=None, showfig=False):
#     from sklearn.neighbors import KernelDensity

#     kde = KernelDensity(kernel=kernel, metric=metric, bandwidth=0.2).fit(X)
#     dens = kde.score_samples(X)

#     # import mpl_scatter_density # adds projection='scatter_density'
#     # pip install mpl-scatter-density
#     # density = plt.scatter_density(x, y, cmap=white_viridis)

#     if showfig:
#         if ax is None: plt.figure(figsize=(8,8))
#         ax.scatter(X[:,0], X[:,1], c=dens)

#     return dens
