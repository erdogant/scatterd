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

#%% Main
def scatterd(X, Y, s=15, c=[0,0,0], label=None, norm=False, cmap='Set1', xlabel=None, ylabel=None, title=None, fontsize=16, fontcolor=None, figsize=(15,10)):
    """Make scaterplot.

    Parameters
    ----------
    X : numpy array
        1D Coordinates.
    Y : numpy array
        1D Coordinates.
    label: list of labels with same size as X
        label of the samples.
    s: Int or list/array of sizes with same size as X
        Size of the samples.
    c: list/array of RGB colors with same size as X
        Color of samples in RGB colors.
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
    xlabel : String, optional
        Xlabel. The default is None.
    ylabel : String, optional
        Ylabel. The default is None.
    title : String, optional
        Title of the figure. The default is None.
    norm : Bool, optional
        Normalize datapoints. The default is True.
    figsize : tuple, optional
        Figure size. The default is (15,10).
    fontsize : String, optional
        The fontsize of the y labels that are plotted in the graph. The default is 16.
    fontcolor: list/array of RGB colors with same size as X (default : None)
        None : Use same colorscheme as for c
        [0,0,0] : If the input is a single color, all fonts will get this color.


    References
    -------
    Colormap: https://matplotlib.org/examples/color/colormaps_reference.html


    Returns
    -------
    Fig, ax

    """
    if len(X)!=len(Y): raise Exception('[scatterd] >ERROR: X should have same length as Y.')
    if s is None: raise Exception('[scatterd] >ERROR: input parameter s(ize) should be not None.')
    if c is None: raise Exception('[scatterd] >ERROR: input parameter c(olors) should be not None.')
    if not isinstance(s, int) and len(s)!=len(X):
        raise Exception('[scatterd] >ERROR: input parameter s(ize) should be of same size of X.')

    args = {}
    args['norm'] = norm
    args['cmap'] = cmap
    args['xlabel'] = xlabel
    args['ylabel'] = ylabel
    args['title'] = title
    args['fontsize'] = fontsize
    args['figsize'] = figsize
    # Color of the axis and grid of the plot
    axiscolor = '#dddddd'

    # Combine into array
    X = np.c_[X, Y]

    # Normalize data
    if args['norm']:
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

    # Create unqiue colors for label
    if label is not None:
        c, _ = colourmap.fromlist(label, cmap=args['cmap'], method='matplotlib')

    # Set fontcolor
    fontcolor = _fontcolor(fontcolor, label, X, args['cmap'])
    # print(fontcolor)

    # Bootup figure
    fig, ax = plt.subplots(figsize=args['figsize'])

    # Scatter
    ax.scatter(X[:,0],X[:,1], facecolor=c, s=s, edgecolor='None')

    # Plot labels
    if label is not None:
        for uilabel in fontcolor.keys():
            XYmean=np.mean(X[label==uilabel,:], axis=0)
            plt.text(XYmean[0],XYmean[1], str(uilabel), color=fontcolor.get(uilabel), fontdict={'weight': 'bold', 'size': args['fontsize']})

    # Labels on axis
    ax.set_xlabel(args['xlabel'])
    ax.set_ylabel(args['ylabel'])
    if args['title'] is not None:
        ax.set_title(args['title'])
        
    # set background to none
    ax.set_facecolor('none')
    # Set grid and axis to grey
    ax.grid(True, color=axiscolor)
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color(axiscolor)

    # Show figure
    fig.show()
    fig.canvas.draw()
    
    # Return
    return(fig,ax)

# %% Fontcolor
def _fontcolor(fontcolor, label, X, cmap, verbose=3):
    # Set font colors
    if (fontcolor is None) and (label is None):
        pass
    elif (fontcolor is not None) and (label is None):
        if verbose>=2: print('[scatterd] >Warning : Without label, there is no font(color) to print.')
    elif (fontcolor is None) and (label is not None):
        _, fontcolor = colourmap.fromlist(label, cmap=cmap, method='matplotlib')
    elif (fontcolor is not None) and (label is not None) and (len(fontcolor)==X.shape[0]):
        _, fontcolor = colourmap.fromlist(fontcolor, cmap=cmap, method='matplotlib')
    elif (fontcolor is not None) and (label is not None) and ( (isinstance(fontcolor[0], int)) or (isinstance(fontcolor[0], float)) ):
        _, tmpcolors = colourmap.fromlist(label, cmap=cmap, method='matplotlib')
        list(map(lambda x: tmpcolors.update({x: fontcolor}), [*tmpcolors.keys()]));
        fontcolor = tmpcolors
    else:
        raise Exception('[scatterd] >ERROR : fontcolor input is not correct.')

    return fontcolor

  