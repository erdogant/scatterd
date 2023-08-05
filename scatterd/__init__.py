from scatterd.scatterd import scatterd, set_colors, _preprocessing, jitter_func, normalize_between_0_and_1
from datazets import get as import_example

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.3.6'

# module level doc-string
__doc__ = """
scatterd
=====================================================================

Scatterd is an easy and fast way of creating beautiful scatter plots.

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
>>> # Only density
>>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=None, density=True)
>>>
>>> # Various settings
>>> fig, ax = scatterd(df['tsneX'], df['tsneY'], labels=df['labx'], density=True, density_on_top=True, args_density={'alpha': 0.3}, gradient='#FFFFFF', edgecolor='#FFFFFF', grid=True, fontweight='normal', fontsize=26, legend=2)
>>>

References
----------
* github: https://github.com/erdogant/scatterd
* Documentation: https://erdogant.github.io/scatterd/
* Colormap: https://matplotlib.org/examples/color/colormaps_reference.html

"""
