"""Scatterplot."""
# --------------------------------------------------
# Name        : scatterd.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/scatterd
# Licence     : See licences
# --------------------------------------------------

# %% Libraries
import colourmap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests
import os
from urllib.parse import urlparse

# %% Main
def scatterd(x,
             y,
             z=None,
             s=150,
             c=[0,0.1,0.4],
             labels=None,
             marker='o',
             alpha=0.8,
             edgecolor='#FFFFFF',
             gradient=None,
             density=False,
             norm=False,
             cmap='tab20c',
             figsize=(25, 15),
             dpi=100,
             legend=True,
             ax=None,
             jitter=None,
             xlabel='x-axis', ylabel='y-axis', title='', fontsize=24, fontcolor=None, grid=False, fontweight='normal',
             args_density = {'cmap': 'Reds', 'fill': True, 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': False},
             visible=True,
             verbose=3):
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
    marker: list/array of strings (default: 'o').
        Marker for the samples.
            * 'x' : All data points get this marker
            * ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X') : Specify per sample the marker type.
            * [0,3,0,1,2,1,..] : It is also possible to provide a list of labels. The markers will be assigned accordingly.
    alpha : float, default: 0.8
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    edgecolors : (default: 'face')
        The edge color of the marker. Possible values:
            * 'face': The edge color will always be the same as the face color.
            * 'none': No patch boundary will be drawn.
            * '#FFFFFF' : A color or sequence of colors.
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
        '#000000' : If the input is a single color, all fonts will get this color.
    grid: str or bool (default: False)
        False or None : Do not plot grid
        True : Set axis color to: '#dddddd'
        '#dddddd' : Specify color with hex
    norm : Bool, optional
        Normalize datapoints. The default is False.
    legend : bool, (default: False)
        Plot the legend.
    jitter : float, default: None
        Add jitter to data points as random normal data. Values of 0.01 is usually good for one-hot data seperation.
            * None or False: Do not add jitter
            * True : adds 0.01
            * 0.05 : Specify the amount jitter to add.
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
    visible : Bool, default: True
        Visible status of the Figure. When False, figure is created on the background.
    args_density : dict()
        Dictionary containing arguments for kernel density plotting.

    Returns
    -------
    Fig, ax

    Examples
    --------
    >>> # Import library
    >>> from scatterd import scatterd, import_example
    >>>
    >>> # Import example
    >>> df = import_example()
    >>>
    >>> # Simple scatter
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'])
    >>>
    >>> # Scatter with labels
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'])
    >>>
    >>> # Scatter with gradient
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], gradient='#FFFFFF')
    >>>
    >>> # Change cmap
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], gradient='#FFFFFF', cmap='Set2')
    >>>
    >>> # Scatter with density
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True)
    >>>
    >>> # Scatter with density and gradient
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True, gradient='#FFFFFF')
    >>>

    References
    -------
    * github: https://github.com/erdogant/scatterd
    * Documentation: https://erdogant.github.io/scatterd/
    * Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

    """
    fig = None
    if len(x)!=len(y): raise Exception('[scatterd] >Error: input parameter x should be the same size of y.')
    # if s is None: raise Exception('[scatterd] >Error: input parameter s(ize) should have value >0.')
    # if c is None: raise Exception('[scatterd] >Error: input parameter c(olors) can not be None.')
    if isinstance(c, str): raise Exception('[scatterd] >Error: input parameter c(olors) should be RGB of type tuple [0,0,0] .')
    if not isinstance(s, int) and len(s)!=len(x): raise Exception('[scatterd] >Error: input parameter s(ize) should be of same size of X.')
    if (z is not None) and len(x)!=len(z): raise Exception('[scatterd] >Error: input parameter z should be the same size of x and y.')
    if s is None: s=0
    if c is None: s, c = 0, [0, 0, 0]

    # Defaults
    defaults_density = {'cmap': 'Reds', 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': False, 'fill': True}
    args_density = {**defaults_density, **args_density}

    # Preprocessing
    X, labels = _preprocessing(x, y, z, labels, jitter, norm)
    # Set color
    c_rgb = set_colors(X, labels, c, cmap, gradient=gradient, verbose=verbose)
    # Set fontcolor
    fontcolor = set_fontcolor(fontcolor, labels, X, cmap, verbose=2)
    # Set size
    s = set_size(X, s)
    # Set size
    alpha = set_alpha(X, alpha)
    # Set marker
    marker = set_marker(X, marker)
    # Bootup figure
    fig, ax = init_figure(ax, z, dpi, figsize, visible)

    # Set figure properties
    ax = _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, grid, fontweight, ax)

    # Add density as bottom layer to the scatterplot
    if density:
        ax = sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, **args_density)
        # ax = sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, hue=labels, **args_density)

    # Scatter all at once
    if (labels is None) and isinstance(marker, str):
        if z is None:
            ax.scatter(X[:, 0], X[:, 1], c=c_rgb, s=s, edgecolor=edgecolor, marker=marker, alpha=alpha)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, c=c_rgb, edgecolor=edgecolor, marker=marker, alpha=alpha)
    else:
        uilabels = np.unique(labels)
        for label in uilabels:
            Iloc1 = label==labels
            # Extract the unique markers
            for m in np.unique(marker[Iloc1]):
                Iloc2 = np.logical_and(Iloc1, m==marker)
                if z is None:
                    ax.scatter(X[Iloc2, 0], X[Iloc2, 1], c=c_rgb[Iloc2], s=s[Iloc2], edgecolor=edgecolor, marker=m, label=label, alpha=alpha[Iloc2])
                else:
                    # Let op: alpha[Iloc2] geeft een foutmelding
                    ax.scatter(X[Iloc2, 0], X[Iloc2, 1], X[Iloc2, 2], s=s[Iloc2], c=c_rgb[Iloc2], edgecolor=edgecolor, marker=m, label=label, alpha=0.8)

    # Show legend (only if labels are present)
    if legend and labels is not None: ax.legend()

    # Return
    return fig, ax


# %% Setup figure properties
def _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, grid, fontweight, ax):
    # Set axis fontsizes
    if grid is None: grid=False
    if grid is True: grid='#dddddd'
    font = {'family': 'DejaVu Sans', 'weight': fontweight, 'size': fontsize}
    matplotlib.rc('font', **font)

    # Plot labels
    if (labels is not None) and (fontcolor is not None):
        for uilabel in fontcolor.keys():
            # Compute median for better center compared to mean
            XYmean = np.median(X[labels==uilabel, :], axis=0)
            if X.shape[1]==2:
                ax.text(XYmean[0], XYmean[1], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': fontweight, 'size': fontsize})
            else:
                ax.text(XYmean[0], XYmean[1], XYmean[2], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': fontweight, 'size': fontsize})

    # Labels on axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    # set background to none
    ax.set_facecolor('none')

    # Set grid and axis to grey
    if isinstance(grid, str):
        ax.grid(True, color=grid)
        # Color all lines in the plot the same grid color
        # for child in ax.get_children():
            # if isinstance(child, matplotlib.spines.Spine):
                # child.set_color(grid)

    return ax


# %% Setup colors
def set_colors(X, labels, c, cmap='tab20c', gradient=None, verbose=3):
    """Set colors."""
    if c is None: return None

    # The default is all dots to black
    c_rgb = np.repeat([0, 0, 0], X.shape[0], axis=0).reshape(-1, 3)

    # Change on input c
    if len(c)==1 and isinstance(c, list): c = c[0]
    if len(c)==3 and isinstance(c[0], (int, float)):
        if verbose>=4: print('[scatterd] >Colors are all set to %s.' %(c))
        c_rgb = np.repeat([c], X.shape[0], axis=0).reshape(-1, 3)

    if X.shape[0]==len(c):
        if verbose>=4: print('[scatterd] >Colors are set to input colors defined in [c].')
        c_rgb=c

    if labels is not None:
        # Create unqiue colors for labels if there are multiple classes or in case cmap and gradient is used.
        if verbose>=4: print('[scatterd] >Colors are based on the input [labels] and on [cmap].')
        if labels is None: labels = np.repeat(0, X.shape[0])
        c_rgb, _ = colourmap.fromlist(labels, cmap=cmap, method='matplotlib', gradient=gradient)

    # Add gradient for each class
    if (gradient is not None):
        if verbose>=4: print('[scatterd] >Color [gradient] is included.')
        c_rgb = gradient_on_density_color(X, c_rgb, labels)

    # Return
    return c_rgb


# %% Create gradient based on density.
def gradient_on_density_color(X, c_rgb, labels):
    """Set gradient on density color."""
    if labels is None: labels = np.repeat(0, X.shape[0])
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
def _preprocessing(x, y, z, labels, jitter, norm=False):
    if jitter is None or jitter is False: jitter=0
    if jitter is True: jitter=0.01
    if jitter>0:
        x = x + np.random.normal(0, jitter, size=len(x))
        if y is not None: y = y + np.random.normal(0, jitter, size=len(y))
        if z is not None: z = z + np.random.normal(0, jitter, size=len(z))

    # Combine into array
    if z is not None:
        X = np.c_[x, y, z]
    else:
        X = np.c_[x, y]
    # Normalize data
    if norm:
        X = _normalize(X)
    # Labels
    # if labels is None:
    #     labels=np.zeros_like(y).astype(int)
    # Return
    return X, labels


# %% Fontcolor
def _normalize(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    return (X - x_min) / (x_max - x_min)


# %% Fontcolor
def set_fontcolor(fontcolor, label, X, cmap, verbose=3):
    if (fontcolor is not None) and colourmap.is_hex_color(fontcolor, verbose=0):
        fontcolor = colourmap.hex2rgb(fontcolor)

    # Set font colors
    if (fontcolor is None) and (label is None):
        # fontcolor = {np.unique(label)[0]: [0, 0, 0]}
        pass
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


def set_size(X, s):
    if isinstance(s, (int, float)): s = np.repeat(s, X.shape[0])
    # Minimum size should be 0 (dots will not be showed)
    s = np.maximum(s, 0)
    return s


def set_alpha(X, alpha):
    if isinstance(alpha, (int, float)): alpha = np.repeat(alpha, X.shape[0])
    # Minimum size should be 0 (dots will not be showed)
    alpha = np.maximum(alpha, 0)
    return alpha


def set_marker(X, marker):
    markers = np.array(['o', 'v', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '.', '^', '>', '<', '8'])
    # markers_list = dict(zip(markers, np.arange(0,len(markers))))
    # In case single str
    if isinstance(marker, str) and np.isin(marker, markers):
        return np.repeat(marker, X.shape[0])
    # In case array of markers.
    if np.all(np.isin(marker, markers)) and len(marker)==X.shape[0]:
        return marker
    # In case array of labels.
    if len(marker)==X.shape[0]:
        d = dict(zip(set(marker), range(len(set(marker)))))
        marker_num = [d[i] for i in marker]
        return markers[np.mod(marker_num, len(markers))]
    # Return
    return np.repeat('o', X.shape[0])

def init_figure(ax, z, dpi, figsize, visible):
    fig = None
    if ax is None:
        # fig, ax = plt.subplots(figsize=figsize)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if z is None:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')

    if fig is not None:
        fig.set_visible(visible)

    return fig, ax


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
# def coord2density(X, kernel='gaussian', metric='euclidean', ax=None, visible=False):
#     from sklearn.neighbors import KernelDensity

#     kde = KernelDensity(kernel=kernel, metric=metric, bandwidth=0.2).fit(X)
#     dens = kde.score_samples(X)

#     # import mpl_scatter_density # adds projection='scatter_density'
#     # pip install mpl-scatter-density
#     # density = plt.scatter_density(x, y, cmap=white_viridis)

#     if visible:
#         if ax is None: plt.figure(figsize=(8,8))
#         ax.scatter(X[:,0], X[:,1], c=dens)

#     return dens
