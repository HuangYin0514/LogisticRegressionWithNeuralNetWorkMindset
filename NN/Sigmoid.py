import numpy as np

def sigmoid(z):
    """
       Compute the sigmoid of z

       Arguments:
       x -- A scalar or numpy array of any size.

       Return:
       s -- sigmoid(z)
       """
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+ np.exp(-z))
    ### END CODE HERE ###

    return s