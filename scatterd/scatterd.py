'''Easy and fast way of creating scatter plots.

 from scatterd import scatterd
 

 Requirements
 ------------
   numpy
   matplotlib
   colourmap


 Name        : scatterd.py
 Author      : E.Taskesen
 Contact     : erdogant@gmail.com
 Date        : Jan. 2020
 Licence     : MIT

 TODO: https://medium.com/@ozan/interactive-plots-with-plotly-and-cufflinks-on-pandas-dataframes-af6f86f62d94
'''

#%% Libraries
import colourmap as colourmap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%% Main
def scatterd(Xcoord, Ycoord, s=15, c=[0,0,0], label=None, norm=False, cmap='Set1', xlabel=None, ylabel=None, title=None, fontsize=16, figsize=(15,10)):
    '''Main function to make scaterplot.


    Parameters
    ----------
    Xcoord : numpy array
        1D Coordinates.

    Ycoord : numpy array
        1D Coordinates.

    label: list of labels with same size as Xcoord
        label of the samples.

    s: Int or list/array of sizes with same size as Xcoord
        Size of the samples.

    c: list/array of RGB colors with same size as Xcoord
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


    References
    -------
    Colormap: https://matplotlib.org/examples/color/colormaps_reference.html


    Returns
    -------
    Fig, ax


    '''
    assert len(Xcoord)==len(Ycoord), print('[SCATTERD] Error. Xcoord should have same length as Ycoord.')
    assert s is not None, print('[SCATTERD] Error. s(ize) should be not None')
    assert c is not None, print('[SCATTERD] Error. c(olors) should be not None')
    if not isinstance(s, int):
        assert len(s)==len(Xcoord), print('[SCATTERD] Error. s should be of same size of Xcoord')

    args=dict()
    args['norm']=norm
    args['cmap']=cmap
    args['xlabel']=xlabel
    args['ylabel']=ylabel
    args['title']=title
    args['fontsize']=fontsize
    args['figsize']=figsize
    # Color of the axis and grid of the plot
    axiscolor='#dddddd'

    # Combine into array
    X=np.c_[Xcoord,Ycoord]

    # Normalize data
    if args['norm']:
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
    
    # Create unqiue colors for label
    if label is not None:
        c,cdict=colourmap.fromlist(label, cmap=args['cmap'], method='matplotlib')

    # Bootup figure
    fig, ax = plt.subplots(figsize=args['figsize'])

    # Scatter
    ax.scatter(X[:,0],X[:,1], facecolor=c, s=s, edgecolor='None')

    # Plot labels
    if label is not None:
        for uilabel in cdict.keys():
            XYmean=np.mean(X[label==uilabel,:], axis=0)
            plt.text(XYmean[0],XYmean[1], str(uilabel), color=cdict.get(uilabel), fontdict={'weight': 'bold', 'size': args['fontsize']})

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


