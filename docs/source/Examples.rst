Quick Scatter
####################################

In the following example we will make a simple scatter plot using all default parameters.

.. code:: python
	
	# Import example iris dataet
	from sklearn import datasets
	iris = datasets.load_iris()
	X = iris.data[:, :2]
	labels = iris.target

	# Load library
	from scatterd import scatterd
	
	# Scatter the results
	fig, ax = scatterd(X[:,0], X[:,1])


.. |fig1| image:: ../figs/fig1_simple.png

.. table:: Basic scatterplot
   :align: center

   +----------+
   | |fig1|   |
   +----------+


Coloring Dots
####################################

Coloring the dots can using RGB values or standard strings, such as 'r', 'k' etc

.. code:: python
	
	# Color dots in red
	fig, ax = scatterd(X[:,0], X[:,1], c=[1,0,0], grid=True)


.. |fig2| image:: ../figs/fig2_red.png

.. table:: Red dots
   :align: center

   +----------+
   | |fig2|   |
   +----------+


Coloring Class Label Fonts
####################################

Coloring the dots can using RGB values or standard strings, such as 'r', 'k' etc

.. code:: python
	
	# Fontcolor in red
	fig, ax = scatterd(X[:,0], X[:,1], edgecolor='k', fontcolor=[0,0,0], fontsize=26)

	# Fontcolor red
	fig, ax = scatterd(X[:,0], X[:,1], edgecolor='k', fontcolor='r', fontsize=26)


.. |fig3| image:: ../figs/fig1_fontcoloring.png

.. table:: Class label coloring
   :align: center

   +----------+
   | |fig3|   |
   +----------+


Coloring on classlabels
####################################

Coloring the dots on the input class labels.

.. code:: python
	
	# Color on classlabels
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, edgecolor='k', fontcolor=[0,0,0], fontsize=26)

	# Change color using the cmap
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, edgecolor='k', fontcolor=[0,0,0], fontsize=26, cmap='Set2')


.. |fig4| image:: ../figs/fig_classlabels1.png
.. |fig5| image:: ../figs/fig_classlabels2.png

.. table:: Class label coloring
   :align: center

   +----------+
   | |fig4|   |
   +----------+
   | |fig5|   |
   +----------+



Overlay with Kernel Density
####################################

Overlay the scatterplot with kernel densities.

.. code:: python
	
	# Add density to plot
	fig, ax = scatterd(X[:,0], X[:,1], density=True)

	# Color the classlabels
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, density=True)

	# Increase dot sizes
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, density=True, s=500)
	
	# Change various parameters
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0])


.. |fig6| image:: ../figs/fig_density_1.png
.. |fig7| image:: ../figs/fig_density_2.png
.. |fig8| image:: ../figs/fig_density_3.png
.. |fig9| image:: ../figs/fig_density_4.png

.. table:: Class label coloring
   :align: center

   +----------+
   | |fig6|   |
   +----------+
   | |fig7|   |
   +----------+
   | |fig8|   |
   +----------+
   | |fig9|   |
   +----------+


Gradient
####################################

Add gradient based on kernel density.
It starts with the color in the highest density will transition towards the gradient color.

.. code:: python
	
	# Add gradient
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, verbose=4, gradient='#ffffff', edgecolor='#ffffff', s=300, cmap='Set1')

	# Add gradient with density
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, verbose=4, gradient='#ffffff', edgecolor='#ffffff', s=300, cmap='Set1', density=True)

	# Add gradient with density and marker but remove the labels
	fig, ax = scatterd(X[:,0], X[:,1], labels=None, marker=labels, verbose=4, gradient='#ffffff', edgecolor='#ffffff', s=300, cmap='Set2', density=True)

	# Add gradient with density and markers and alpha
	import matplotlib as mpl
	custom_cmap = mpl.colors.ListedColormap(['green', 'black', 'blue'])
	s = (labels+1) * 200
	random_integers = np.random.randint(0, len(s), size=X.shape[0])
	alpha = np.random.rand(1, X.shape[0])[0][random_integers]

	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, marker=labels, gradient='#ffffff', edgecolor='#ffffff', s=s, density=True, alpha=alpha, cmap=custom_cmap)


.. |fig11| image:: ../figs/fig_gradient_11.png
.. |fig12| image:: ../figs/fig_gradient_12.png
.. |fig13| image:: ../figs/fig_gradient_13.png
.. |fig14| image:: ../figs/fig_gradient_14.png

.. table:: Class label coloring
   :align: center

   +----------+----------+
   | |fig11|  | |fig12|  |
   +----------+----------+
   | |fig13|  | |fig14|  |
   +----------+----------+


Customized colormap
####################################

Overlay the scatterplot with kernel densities.

.. code:: python
	
	# Change various parameters
	args_density = {'fill':True, 'thresh': 0, 'levels': 100, 'cmap':"mako"}

	# Scatter
	fig, ax = scatterd(X[:,0], X[:,1], labels=labels, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, density=True, fontcolor=[0,0,0], grid=None, args_density=args_density)



.. |fig10| image:: ../figs/custom_args.png

.. table:: Custom colormap
   :align: center

   +----------+
   | |fig10|  |
   +----------+




.. include:: add_bottom.add