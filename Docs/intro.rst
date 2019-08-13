What is Foresight?
------------------

Foresight is a collection of tools built to forecast the future price movements of industrial metals, using Long Short Term Memory networks. It can take univariate or multivariate datasets and make predictions using single-input single-output (SISO), multi-input single-output (MISO) or multi-input multi-output (MIMO) frameworks. 

It was built for the purpose of testing the hypothesis that improved predictive performance can be achieved by applying the multi-task learning paradigm to commodity price forecasting. As such many of the example notebooks are built for this purpose.

The tools can equally be applied to any user chosen datasets, provided the datasets are loaded in the format shown in the example csvs, or are inputed directly as shown in the Proof of Concept (POC) notebooks.


Installation
------------

To install
::
	git clone https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom.git


Requirements
------------

There are package dependancies on the following files:

- numpy==1.16.2
- pandas==0.24.2
- pmdarima==1.2.1
- matplotlib==3.0.3
- psycopg2==2.7.6.1
- scikit-learn==0.20.3
- statsmodels==0.9.0
- torch==1.1.0

Examples and Usage
------------------

Generic regression examples for univariate and multi-variate problems are contained within the POC (proof of concept) notebooks. 

For examples relating to industrial metal forecasting, univariate and multivariate examples can be found in the LSTM notebook, while a multi-task learning example can be found in the LSTM MTL notebook.