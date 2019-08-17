# Foresight [![Build Status](https://travis-ci.com/msc-acse/acse-9-independent-research-project-OliverJBoom.svg?branch=master)](https://travis-ci.com/msc-acse/acse-9-independent-research-project-OliverJBoom) [![Documentation Status](https://readthedocs.org/projects/industrial-metals-forecaster/badge/?version=latest)](https://industrial-metals-forecaster.readthedocs.io/en/latest/?badge=latest)

## Introduction

Foresight is a collection of tools built in Python 3 to forecast the future price movements of industrial metals, using Long Short Term Memory networks. It can take univariate or multivariate datasets and make predictions using single-input single-output (SISO), multi-input single-output (MISO) or multi-input multi-output (MIMO) frameworks.

It was built for the purpose of testing the hypothesis that improved predictive performance can be achieved by applying the multi-task learning paradigm to commodity price forecasting. As such many of the example notebooks are built for this purpose.

The tools can equally be applied to any user chosen datasets, provided the datasets are loaded in the format shown in the example csvs, or are inputed directly as shown in the "generic" notebooks.

## Documentation

Full documentation can be found on https://industrial-metals-forecaster.readthedocs.io/en/latest/.

Read the docs will not show the models module online. For more information on the models module refer to the [PDF Documentation](https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom/blob/master/Docs/_build/latex/IndustrialMetalsForecaster.pdf)<sup>1</sup>

## Repository Structure

* __Docs__                          -  Contains all files relating to the documentation

* __Foresight__                    -  Contains the collection of python files relating to Foresight

* __Foresight/Data/Commodity_Data__ -  Contains the time series used to forecast industrial metals prices

* __Foresight/Notebooks__          -  Contains example notebooks for SISO, MISO and MIMO regression

* __Foresight/Results/Pths__        -  Contains saved model weights

* __Foresight/Results/Pickles__     -  Contains saved deeplearning objects

* __Foresight/Results/Plots__       -  Contains images of predictions and training/val losses 

## Testing

Continuous integration best practices has been utilized, using Travis CL. With the tests methodology used being unit testing written deveoped using the Pytest library.

## Relevant Author and Course Information 

__Author:__ Oliver Boom
__Github:__ OliverJBoom
__CID:__ 01593306

This collection of tools was built as part of the Applied Computational Science & Engineering MSc, completed at Imperial College London. It forms a composite part of the Independent Research Project (Module Code: ASCE9) and was conducted under the supervision of Dr Stephen Neethling. This project is also understaken in partnership with Commodities AI (ChAI), under the supervision of Dr Tristan Fletcher. 

With the exception of parts of the ChaiDB class (the init, get_instrument_data, close_db_connection and get_list_datascope_instruments functions), all of the work contained within this repository is my own.


## License

Foresight is licensed under the MIT license. See [LICENSE](https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom/blob/master/LICENSE).


<sup>1</sup>: Pytorch is too heavy a dependency for Read the Docs Docker containers. And so requires mocking when passing through the Sphinx autodocumentation. However, inheritance from mocked classes is not supported. It is a known issue in Read the Docs. 
