# scatterd

[![Python](https://img.shields.io/pypi/pyversions/scatterd)](https://img.shields.io/pypi/pyversions/scatterd)
[![PyPI Version](https://img.shields.io/pypi/v/scatterd)](https://pypi.org/project/scatterd/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/scatterd/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/scatterd.svg)](https://github.com/erdogant/scatterd/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/scatterd.svg)](https://github.com/erdogant/scatterd/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/scatterd/month)](https://pepy.tech/project/scatterd/month)
[![Downloads](https://pepy.tech/badge/scatterd)](https://pepy.tech/project/scatterd)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

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


### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

### Licence
See [LICENSE](LICENSE) for details.
