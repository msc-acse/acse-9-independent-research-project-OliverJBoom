
.. role:: python(code)
   :language: python


Introduction
============


What is Foresight?
------------------

Foresight is a collection of tools built to forecast the future price movements of industrial metals, using Long Short Term Memory networks. It can take univariate or multivariate datasets and make predictions using single-input single-output (SISO), multi-input single-output (MISO) or multi-input multi-output (MIMO) frameworks. 

It was built for the purpose of testing the hypothesis that improved predictive performance can be achieved by applying the multi-task learning paradigm to commodity price forecasting. As such many of the example notebooks are built for this purpose.

The tools can equally be applied to any user chosen datasets, provided the datasets are loaded in the format shown in the example csvs, or are inputed directly as shown in the "generic" notebooks.


Installation
------------

To install:
::
    pip install ForesightPy

Ensure all requirements in requirements.txt are installed.

Most requirements can be installed by using  :python:`pip install -r requirements.txt`.
Except Pytorch, which requires a more specific installation procedure. This can be found on https://pytorch.org/get-started/locally/.

Example notebooks and datasets are contained within the source repo. This can be downloaded using the following:
::
    git clone https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom.git

To import package
::
    from ForesightPy import *

Requirements
------------

There are package dependancies on the following files:

- numpy>=1.16.2
- pandas>=0.24.2
- pmdarima>=1.2.1
- matplotlib>=3.0.3
- scikit-learn>=0.20.3
- statsmodels>=0.9.0
- torch>=1.1.0

Examples and Usage
------------------

All examples can be found within the Notebooks folder.

Generic regression examples for univariate and multi-variate problems are contained within the "generic" notebooks. 

For examples relating to industrial metal price forecasting; univariate, multivariate and multi-task examples can be found in the "metals forecaster" notebooks.

Hyperparameter Tuning
---------------------

Python files can be found in the Tuning folder of the repository which can be used to investigate the effects of changing hyper parameters. This can be extended to grid search in n dimensions by adding n for loops, to search through the parameter design space.