#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    params[ofs:ofs+ Dx * H]
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Forward pass
    # M = int(data.shape[0])
    # forward propagation
    a = np.matmul(data, W1) + b1  # np.tile(b1, (M, 1))
    h = sigmoid(a)
    theta = np.matmul(h, W2) + b2  # np.tile(b2, (M, 1))
    output = softmax(theta)

    # Compute the probability
    return output[0][label]


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    # print W1.shape, b1.shape, W2.shape, b2.shape

    M = int(data.shape[0])
    # forward propagation
    a = np.matmul(data, W1) + np.tile(b1, (M, 1))
    h = sigmoid(a)
    theta = np.matmul(h, W2) + np.tile(b2, (M, 1))
    output = softmax(theta)
    log_output = np.log(output)
    cost = np.sum(-(log_output * labels))

    # backward propagation
    helper = output - labels
    gradb2 = np.sum(helper, 0)
    gradW2 = np.sum(np.matmul(np.expand_dims(h, 2), np.expand_dims(helper, 1)), 0)
    nabla_j = np.matmul(helper, W2.transpose()) * h * (1 - h)
    gradb1 = np.sum(nabla_j, 0)
    gradW1 = np.sum(np.matmul(np.expand_dims(data, 2), np.expand_dims(nabla_j, 1)), 0)

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    """print "shapes"
    print gradW2.shape, W2.shape, h.shape, helper.shape
    print gradb2.shape, b2.shape
    print gradW1.shape, W1.shape, data.shape, nabla_j.shape
    print gradb1.shape, b1.shape"""

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    pass

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
