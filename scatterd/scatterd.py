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
import requests
import os
from urllib.parse import urlparse

# %% Main
def scatterd(x, y, z=None, s=50, c=[0, 0, 0], labels=None, marker=None, alpha=None, gradient=None, density=False, norm=False, cmap='Set1', figsize=(25, 15), legend=True, ax=None,
             xlabel='x-axis', ylabel='y-axis', title='', fontsize=24, fontcolor=None, axiscolor='#dddddd',
             args_density = {'cmap': 'Reds', 'shade': True, 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': False, 'fill': True}):
    """Make scaterplot.

    Parameters
    ----------
    x : numpy array
        1D Coordinates x-axis.
    y : numpy array
        1D Coordinates y-axis.
    z : numpy array
        1D Coordinates z-axis.
    s: Int or list/array of sizes with same size as X
        Size of the samples.
    c: list/array of RGB colors with same size as X
        Color of samples in RGB colors.
    labels: list of labels with same size as X
        labels of the samples.
    gradient : String, (default: None)
        Hex color to make a lineair gradient for the scatterplot.
        '#FFFFFF'
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

    Returns
    -------
    Fig, ax

    Examples
    --------
    >>> # Import library
    >>> from scatterd import scatterd, import_example
    >>> # Import example
    >>> df = import_example()
    >>> # plain scatter plot
    >>> ax = scatterd(df['tsneX'], df['tsneY'])
    >>> # plain scatter plot
    >>> ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'])
    >>> # Gradient
    >>> ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], gradient='#FFFFFF')
    >>> # Change cmap
    >>> ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], gradient='#FFFFFF', cmap='Set2')
    >>> # Density
    >>> ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True)
    >>> # Density with gradient
    >>> ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True, gradient='#FFFFFF')
    >>>

    References
    -------
    Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

    """
    fig = None
    if len(x)!=len(y): raise Exception('[scatterd] >Error: input parameter x should be the same size of y.')
    if s is None: raise Exception('[scatterd] >Error: input parameter s(ize) should have value >0.')
    if c is None: raise Exception('[scatterd] >Error: input parameter c(olors) should be not None.')
    if isinstance(c, str): raise Exception('[scatterd] >Error: input parameter c(olors) should be RGB of type tuple [0,0,0] .')
    if not isinstance(s, int) and len(s)!=len(x): raise Exception('[scatterd] >Error: input parameter s(ize) should be of same size of X.')
    if (z is not None) and len(x)!=len(z): raise Exception('[scatterd] >Error: input parameter z should be the same size of x and y.')

    # Defaults
    defaults_kde = {'cmap': 'Reds', 'shade': True, 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': False, 'fill': True}
    args_density = {**defaults_kde, **args_density}

    # Preprocessing
    X, labels = _preprocessing(x, y, z, labels, norm)

    # Figure properties
    c_rgb, fontcolor = set_colors(X, labels, fontcolor, c, cmap, gradient=gradient)

    if s is None:
        s = np.repeat(50, X.shape[0])
    elif isinstance(s, int):
        s = np.repeat(s, X.shape[0])

    # Bootup figure
    if ax is None:
        # fig, ax = plt.subplots(figsize=figsize)
        fig = plt.figure(figsize=figsize)
        if z is None:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')

    uilabels = np.unique(labels)
    for label in uilabels:
        Iloc = label==labels
        if z is None:
            ax.scatter(X[Iloc, 0], X[Iloc, 1], color=c_rgb[Iloc], s=s[Iloc], edgecolor='None', marker=marker, label=label)
        else:
            ax.scatter(X[Iloc, 0], X[Iloc, 1], X[Iloc, 2], s=s[Iloc], color=c_rgb[Iloc], edgecolor='None', marker=marker, label=label)

    # Add density to scatterplot
    if density:
        ax = sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, **args_density)
        # ax = sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, hue=labels, **args_density)

    # Set figure properties
    ax = _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, axiscolor, ax)

    # Show figure
    if legend: ax.legend()

    # Return
    return fig, ax


# %% Setup figure properties
def _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, axiscolor, ax):
    # Set axis fontsizes
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    # Plot labels
    if (labels is not None) and (fontcolor is not None):
        for uilabel in fontcolor.keys():
            XYmean = np.mean(X[labels==uilabel, :], axis=0)
            if X.shape[1]==2:
                ax.text(XYmean[0], XYmean[1], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': 'bold', 'size': fontsize})
            else:
                ax.text(XYmean[0], XYmean[1], XYmean[2], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': 'bold', 'size': fontsize})

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
def set_colors(X, labels, fontcolor, c, cmap, gradient=None):

    # Create unqiue colors for labels
    if len(np.unique(labels))>1:
        c_rgb, _ = colourmap.fromlist(labels, cmap=cmap, method='matplotlib', gradient=gradient)
    elif len(c)==3 and (isinstance(c[0], int) or isinstance(c[0], float)):
        c_rgb = np.repeat([c], X.shape[0], axis=0).reshape(-1, 3)
    elif len(c)==1 and isinstance(c, list):
        c_rgb = np.repeat(c[0], X.shape[0], axis=0).reshape(-1, 3)
    else:
        c_rgb = np.repeat([0, 0, 0], X.shape[0], axis=0).reshape(-1, 3)

    if (gradient is not None):
        c_rgb = gradient_on_density_color(X, c_rgb, labels)

    # Set fontcolor
    fontcolor = _fontcolor(fontcolor, labels, X, cmap)
    # Return
    return c_rgb, fontcolor


# %% Create gradient based on density.
def gradient_on_density_color(X, c_rgb, labels):
    from scipy.stats import gaussian_kde
    uilabels = np.unique(labels)
    density_colors= np.repeat([1., 1., 1.], len(labels), axis=0).reshape(-1, 3)

    if (len(uilabels)!=len(labels)):
        for label in uilabels:
            idx = np.where(labels==label)[0]
            if X.shape[1]==2:
                xy = np.vstack([X[idx, 0], X[idx, 1]])
            else:
                xy = np.vstack([X[idx, 0], X[idx, 1], X[idx, 2]])

            try:
                # Compute density
                z = gaussian_kde(xy)(xy)
                # Sort on density
                didx = idx[np.argsort(z)[::-1]]
            except:
                didx=idx

            # order colors correctly based Density
            density_colors[didx] = c_rgb[idx, :]
            # plt.figure()
            # plt.scatter(X[didx,0], X[didx,1], color=c_rgb[idx, :])
            # plt.figure()
            # plt.scatter(idx, idx, color=c_rgb[idx, :])
        c_rgb=density_colors

    # Return
    return c_rgb


# %% Fontcolor
def _preprocessing(x, y, z, labels, norm=False):
    # Combine into array
    if z is not None:
        X = np.c_[x, y, z]
    else:
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


# %% Import example dataset from github.
def import_example(data='cancer', url=None, sep=',', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'cancer'
    url : str
        url link to to dataset.
    verbose : int, (default: 3)

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    import pandas as pd

    if url is None:
        url='https://erdogant.github.io/datasets/cancer_dataset.zip'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        if verbose>=3: print('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('Downloading [%s] dataset from github source..' %(data))
        wget(url, PATH_TO_DATA)

    # Import local dataset
    if verbose>=3: print('Import dataset [%s]' %(data))

    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df


# %% Download files from github source
def wget(url, writepath):
    """Retrieve file from url.

    Parameters
    ----------
    url : str.
        Internet source.
    writepath : str.
        Directory to write the file.

    Returns
    -------
    None.

    Example
    -------
    >>> import clustimage as cl
    >>> images = cl.wget('https://erdogant.github.io/datasets/flower_images.zip', 'c://temp//flower_images.zip')

    """
    r = requests.get(url, stream=True)
    with open(writepath, "wb") as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)

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
