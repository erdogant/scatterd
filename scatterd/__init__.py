from scatterd.scatterd import scatterd,import_example, set_colors, _preprocessing, gradient_on_density_color

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.2.2'

# module level doc-string
__doc__ = """
scatterd
=====================================================================

Description
-----------
Scatterd is an easy and fast way of creating scatter plots.

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
----------
* github: https://github.com/erdogant/scatterd
* Documentation: https://erdogant.github.io/scatterd/
* Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

"""
