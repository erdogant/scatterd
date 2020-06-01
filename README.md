# scatterd

[![Python](https://img.shields.io/pypi/pyversions/scatterd)](https://img.shields.io/pypi/pyversions/scatterd)
[![PyPI Version](https://img.shields.io/pypi/v/scatterd)](https://pypi.org/project/scatterd/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/scatterd/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/scatterd/week)](https://pepy.tech/project/scatterd/week)
[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* Easy and fast manner of creating scatter plots.

## Contents
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install scatterd from PyPI (recommended). scatterd is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

## Quick Start
```
pip install scatterd
```

* Alternatively, install scatterd from the GitHub source:
```bash
git clone https://github.com/erdogant/scatterd.git
cd scatterd
python setup.py install
```  

### Import scatterd package
```python
from scatterd import scatterd
```

### Example:
```python
# Import some example data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Make simple scatterplot
scatterd(X[:,0], X[:,1])
# Color based on labels
scatterd(X[:,0], X[:,1], label=y, s=100)
# Set labels
scatterd(X[:,0], X[:,1], label=y, s=100, norm=True, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title')
# Change sizes
s=np.random.randint(10,200,len(y))
scatterd(X[:,0], X[:,1], label=y, s=s, cmap='Set2', xlabel='xlabel', ylabel='ylabel', title='Title', fontsize=25, figsize=(15,10))
# Change figure size
scatterd(X[:,0], X[:,1], figsize=(25,15))

```
<p align="center">
  <img src="https://github.com/erdogant/scatterd/blob/master/docs/figs/fig1.png" width="600" />
  <img src="https://github.com/erdogant/scatterd/blob/master/docs/figs/fig2.png" width="600" />
  <img src="https://github.com/erdogant/scatterd/blob/master/docs/figs/fig3.png" width="600" />
  <img src="https://github.com/erdogant/scatterd/blob/master/docs/figs/fig4.png" width="600" />
</p>


### Citation
Please cite scatterd in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019scatterd,
  title={scatterd},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/scatterd}},
}
```

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
