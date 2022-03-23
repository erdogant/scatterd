from scatterd.scatterd import scatterd,import_example, set_colors, _preprocessing, gradient_on_density_color

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.1.1'

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
----------
https://github.com/erdogant/scatterd

"""
