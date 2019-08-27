Deeplearning Module
===================

Here are contained the set of functions relating to the training,
validation and testing of the neural networks.

If the user intends to load pickles of saved DeepLearning objects or model pth
files it is important to remember that the models must be loaded in the same
computational environment as they were initialised in. Both in terms of
parallelisation and the processing units they are loaded on.

For example if a model was trained on 16 GPUs in parallel, it will be required
that that model is loaded on 16 GPUs in parallel. This is a pre-requisite
required by Pytorch in their serialization routines.

.. automodule:: Foresight.deeplearning
    :members:
