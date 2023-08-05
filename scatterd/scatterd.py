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
import logging

logger = logging.getLogger('')
[logger.removeHandler(handler) for handler in logger.handlers[:]]
console = logging.StreamHandler()
formatter = logging.Formatter('[scatterd] >%(levelname)s> %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger = logging.getLogger(__name__)


# %% Main
def scatterd(x,
             y,
             z=None,
             s=150,
             c=[0, 0.1, 0.4],
             labels=None,
             marker='o',
             alpha=0.8,
             edgecolor='#000000',
             gradient='opaque',
             opaque_type='per_class',
             density=True,
             density_on_top=False,
             norm=False,
             cmap='tab20c',
             figsize=(25, 15),
             dpi=100,
             legend=None,
             jitter=None,
             xlabel='x-axis', ylabel='y-axis', title='', fontsize=24, fontcolor=None, grid=True, fontweight='normal',
             args_density = {'cmap': 'Reds', 'fill': True, 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': False},
             visible=True,
             fig=None,
             ax=None,
             verbose='info'):
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
    gradient : String, (default: 'opaque')
        Hex color to make a lineair gradient using the density.
            * None: Do not use gradient.
            * opaque: Towards the edges the points become more opaque and thus not visible.
            * '#FFFFFF': Towards the edges it smooths into this color
    opaque_type : String, optional
            * 'per_class': Transprancy is determined on the density within the class label (y)
            * 'all': Transprancy is determined on all available data points
            * 'lineair': Transprancy is lineair set within the class label (y)
    density : Bool (default: False)
        Include the kernel density in the scatter plot.
    density_on_top : bool, (default: False)
        True : The density is the highest layer.
        False : The density is the lowest layer.
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
    legend : int, default: 0
        None: Set automatically based number of labels.
        False : Disable.
        True : Best position.
        1 : 'upper right'
        2 : 'upper left'
        3 : 'lower left'
        4 : 'lower right'
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
    verbose : int, (default: 'info')
        Print progress to screen. The default is 'info'
        60: None, 40: error, 30: warning, 20: info, 10: debug

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
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], edgecolor='#FFFFFF')
    >>>
    >>> # Scatter with labels
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'])
    >>>
    >>> # Scatter with labels
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'])
    >>>
    >>> # Scatter with gradient
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], z=df['tsneY'], labels=df['labx'], gradient='#FFFFFF')
    >>>
    >>> # Change cmap
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], gradient='#FFFFFF', cmap='Set2')
    >>>
    >>> # Scatter with density
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True)
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=False, gradient='#FFFFFF', edgecolor='#FFFFFF')
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True, gradient='#FFFFFF', edgecolor='#FFFFFF')
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True, gradient='#FFFFFF', c=None)
    >>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True, density_on_top=True, args_density={'alpha': 0.3}, gradient='#FFFFFF', edgecolor='#FFFFFF', grid=True, fontweight='normal', fontsize=26, legend=2)
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
    # Set the logger
    set_logger(verbose=verbose)

    if len(x)!=len(y): raise Exception('[scatterd] >Error: input parameter x should be the same size of y.')
    if isinstance(c, str): raise Exception('[scatterd] >Error: input parameter c(olors) should be RGB of type tuple [0,0,0] .')
    if not isinstance(s, int) and len(s)!=len(x): raise Exception('[scatterd] >Error: input parameter s(ize) should be of same size of X.')
    if (z is not None) and len(x)!=len(z): raise Exception('[scatterd] >Error: input parameter z should be the same size of x and y.')
    if s is None: s=0
    if c is None: s, c = 0, [0, 0, 0]
    if isinstance(s, (int, float)) and s==0: fontsize=0
    zorder = None if density_on_top else 10

    # Defaults
    defaults_density = {'cmap': 'Reds', 'thresh': 0.05, 'bw_adjust': .6, 'alpha': 0.66, 'legend': False, 'cbar': False, 'fill': True}
    args_density = {**defaults_density, **args_density}

    # Preprocessing
    X, labels = _preprocessing(x, y, z, labels, jitter, norm)
    # Set color
    c_rgb, opaque = set_colors(X, labels, c, cmap, gradient=gradient, opaque_type=opaque_type)
    # Set fontcolor
    fontcolor = set_fontcolor(fontcolor, labels, X, cmap)
    # Set size
    s = set_size(X, s)
    # Set size
    alpha = set_alpha(X, alpha, gradient, opaque)
    # Set marker
    marker = set_marker(X, marker)
    # Bootup figure
    fig, ax = init_figure(ax, z, dpi, figsize, visible, fig)
    # Set figure properties
    ax = _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, grid, fontweight, zorder, ax)

    # Add density as bottom layer to the scatterplot
    if density:
        logger.info('Add density layer')
        ax = sns.kdeplot(x=X[:, 0], y=X[:, 1], ax=ax, **args_density)

    # Scatter all at once
    if (labels is None) and isinstance(marker, str):
        logger.info('Create scatterplot (fast)')
        # Do not try to plot legend.
        legend = -1
        if z is None:
            ax.scatter(X[:, 0], X[:, 1], c=c_rgb, s=s, edgecolor=edgecolor, marker=marker, alpha=alpha, zorder=zorder)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, c=c_rgb, edgecolor=edgecolor, marker=marker, alpha=alpha, zorder=zorder)
    else:
        logger.info('Create scatterplot')
        uilabels = np.unique(labels)
        for label in uilabels:
            Iloc1 = label==labels
            # Extract the unique markers
            for m in np.unique(marker[Iloc1]):
                Iloc2 = np.logical_and(Iloc1, m==marker)
                if z is None:
                    ax.scatter(X[Iloc2, 0], X[Iloc2, 1], c=c_rgb[Iloc2], s=s[Iloc2], edgecolor=edgecolor, marker=m, label=label, alpha=alpha[Iloc2], zorder=zorder)
                else:
                    # Note: alpha[Iloc2] throws an error
                    ax.scatter(X[Iloc2, 0], X[Iloc2, 1], X[Iloc2, 2], s=s[Iloc2], c=c_rgb[Iloc2], edgecolor=edgecolor, marker=m, label=label, alpha=0.8, zorder=zorder)

    # Show legend (only if labels are present)
    if isinstance(legend, bool): legend = 0 if legend else -1
    if legend is None: legend = -1 if len(np.unique(labels))>20 else 0
    if legend>=0: ax.legend(loc=legend, fontsize=14)

    # Return
    return fig, ax


# %% Setup figure properties
def _set_figure_properties(X, labels, fontcolor, fontsize, xlabel, ylabel, title, grid, fontweight, zorder, ax):
    # Set axis fontsizes
    if grid is None: grid=False
    if grid is True: grid='#dddddd'
    None if zorder is None else zorder + 1
    font = {'family': 'DejaVu Sans', 'weight': fontweight, 'size': fontsize}
    matplotlib.rc('font', **font)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    # Plot labels
    if (labels is not None) and (fontcolor is not None):
        for uilabel in fontcolor.keys():
            # Compute median for better center compared to mean
            XYmean = np.mean(X[labels==uilabel, :], axis=0)
            if X.shape[1]==2:
                ax.text(XYmean[0], XYmean[1], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': fontweight, 'size': fontsize}, zorder=zorder)
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
def set_colors(X, labels, c, cmap='tab20c', gradient=None, opaque_type='per_class', verbose=None):
    """Set colors."""
    # Set the logger
    if verbose is not None: set_logger(verbose=verbose)
    if c is None: return None

    # The default is all dots to black
    c_rgb = np.repeat([0, 0, 0], X.shape[0], axis=0).reshape(-1, 3)
    # Create opaque levels
    opaque = np.array([1.0] * X.shape[0])

    # Change on input c
    if len(c)==1 and isinstance(c, list): c = c[0]
    if len(c)==3 and isinstance(c[0], (int, float)):
        logger.debug('Colors are all set to %s.' %(c))
        c_rgb = np.repeat([c], X.shape[0], axis=0).reshape(-1, 3)

    if X.shape[0]==len(c):
        logger.debug('Colors are set to input colors defined in [c].')
        c_rgb=c

    if labels is not None:
        # Create unqiue colors for labels if there are multiple classes or in case cmap and gradient is used.
        logger.debug('Colors are based on the input [labels] and on [cmap].')
        if labels is None: labels = np.repeat(0, X.shape[0])
        c_rgb = colourmap.fromlist(labels, X=X, cmap=cmap, scheme='rgb', method='matplotlib', gradient=gradient, opaque_type=opaque_type, verbose=verbose)[0]
        # Add extra column with transparancy
        if gradient=='opaque' and c_rgb.shape[1]==4:
            # Set the minimum transparancy level at 0.1
            opaque = np.maximum(c_rgb[:, -1], 0.1)
            c_rgb = c_rgb[:, :3]

    # Return
    return c_rgb, opaque


# %% Jitter
def jitter_func(x, jitter=0.01):
    """Add jitter to data.

    Noise is generated from random normal distribution and added to the data.

    Parameters
    ----------
    x : numpy array
        input data.
    jitter : float, optional
        Strength of generated noise. The default is 0.01.

    Returns
    -------
    x : array-like
        Data including noise.

    """
    if jitter is None or jitter is False: jitter=0
    if jitter is True: jitter=0.01
    if jitter>0 and x is not None:
        x = x + np.random.normal(0, jitter, size=len(x))
    return x


# %% Fontcolor
def _preprocessing(x, y, z, labels, jitter, norm=False):
    # Add jitter
    x = jitter_func(x, jitter=jitter)
    y = jitter_func(y, jitter=jitter)
    z = jitter_func(z, jitter=jitter)

    # Combine into array
    if z is not None:
        X = np.c_[x, y, z]
    else:
        X = np.c_[x, y]
    # Normalize data
    if norm:
        X = normalize_between_0_and_1(X)
    # Labels
    # if labels is None:
    #     labels=np.zeros_like(y).astype(int)
    # Return
    return X, labels


# %% Normalize between [0-1]
def normalize_between_0_and_1(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    out = (X - x_min) / (x_max - x_min)
    out[np.isnan(out)]=1
    return out


# %% Fontcolor
def set_fontcolor(fontcolor, label, X, cmap, verbose=None):
    # Set the logger
    if verbose is not None: set_logger(verbose=verbose)
    if (fontcolor is not None) and colourmap.is_hex_color(fontcolor, verbose=get_logger()):
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
        logger.warning('Without label, there is no font(color) to print.')
    elif (fontcolor is None) and (label is not None):
        _, fontcolor = colourmap.fromlist(label, cmap=cmap, method='matplotlib', verbose=get_logger())
    elif (fontcolor is not None) and (label is not None) and (len(fontcolor)==X.shape[0]):
        _, fontcolor = colourmap.fromlist(fontcolor, cmap=cmap, method='matplotlib', verbose=get_logger())
    elif (fontcolor is not None) and (label is not None) and ((isinstance(fontcolor[0], int)) or (isinstance(fontcolor[0], float))):
        _, tmpcolors = colourmap.fromlist(label, cmap=cmap, method='matplotlib', verbose=get_logger())
        # list(map(lambda x: tmpcolors.update({x: fontcolor}), [*tmpcolors.keys()]))
        fontcolor = tmpcolors
    else:
        raise Exception('[scatterd] >ERROR : fontcolor input is not correct.')

    return fontcolor


def set_size(X, s):
    """Set size."""
    if isinstance(s, (int, float)): s = np.repeat(s, X.shape[0])
    # Minimum size should be 0 (dots will not be showed)
    s = np.maximum(s, 0)
    return s


def set_alpha(X, alpha, gradient, opaque, verbose=None):
    """Set alpha."""
    # Set the logger
    if verbose is not None: set_logger(verbose=verbose)
    if alpha is None: alpha=0.8
    if isinstance(alpha, (int, float)):
        alpha = np.repeat(alpha, X.shape[0])
    if gradient=='opaque':
        logger.info('Set alpha based on density because of the parameter: [%s]' %(gradient))
        alpha = opaque
    # Minimum size should be 0 (dots will not be showed)
    alpha = np.maximum(alpha, 0)
    return alpha


def set_marker(X, marker):
    """Set markers."""
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


def init_figure(ax, z, dpi, figsize, visible, fig):
    """Initialize figure."""
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if z is None:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')
            # Rotate
            ax.view_init(-140, 60)

    if fig is not None:
        fig.set_visible(visible)

    return fig, ax


# %%
def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[scatterd]> WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose


# %%
def get_logger():
    return logger.getEffectiveLevel()


# %%
def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical']: Messages from critical level and higher.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert to verbose
    verbose = convert_verbose_to_new(verbose)
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'error': 50,
                  'critical': 50}
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)
    logger.debug('Set verbose to %s' %(verbose))


# %%
def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


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
