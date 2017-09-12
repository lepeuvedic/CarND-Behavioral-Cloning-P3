#!/bin/bash
# Run jupyter notebook in proper environment.
# Note that model.h5 is saved in the same environment, and for making predictions, theano should be used
# either on CPU or on GPU. For training a GPU is necessary.
KERAS_BACKEND=theano
export KERAS_BACKEND
DEVICE=opencl0:0
export DEVICE

# Required for simulator compatibility (csv file and client-server)
LC_ALL=C
export LC_ALL

