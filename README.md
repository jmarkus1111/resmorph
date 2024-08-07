# resmorph

A Python package acting as a wrapper around the ``galight`` and ``lenstronomy`` fitting package that presents a pipeline to quantify the morphology of high-redshift galaxies using Sersic residuals. ``resmorph`` provides functions for automatically fitting Sersic profiles to large samples of galaxies and calculating a "residual score" for each galaxy. The residual score represents how accurately the light distribution of a galaxy and its surroundings can be represented with Sersic profiles, and can serve as a rough quantification of the galaxy's morphology. 

Installation
------------
    $ pip install -i https://test.pypi.org/simple/ resmorph

Getting Started
---------------
[Example notebook](https://github.com/jmarkus1111/resmorph/blob/main/example_notebook.ipynb)

Requirements
---------------
Tested and recomended with Python == 3.9.19, as well as these [further package requirements](https://github.com/jmarkus1111/resmorph/blob/main/requirements.txt).
  
Citation
--------
