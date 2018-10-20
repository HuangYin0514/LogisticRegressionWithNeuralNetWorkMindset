import numpy as np


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
       This function optimizes w and b by running a gradient descent algorithm

       Arguments:
       w -- weights, a numpy array of size (num_px * num_px * 3, 1)
       b -- bias, a scalar
       X -- data of shape (num_px * num_px * 3, number of examples)
       Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
       num_iterations -- number of iterations of the optimization loop
       learning_rate -- learning rate of the gradient descent update rule
       print_cost -- True to print the loss every 100 steps

       Returns:
       params -- dictionary containing the weights w and bias b
       grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
       costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

       Tips:
       You basically need to write down two steps and iterate through them:
           1) Calculate the cost and the gradient for the current parameters. Use propagate().
           2) Update the parameters using gradient descent rule for w and b.
       """
    nw = w
    nb = b
    cost = np.zeros(shape=(num_iterations,1))

    for i in range(num_iterations):
        from Propagate import propagate
        grads, cost = propagate(nw, nb, X, Y)
        nw = nw - learning_rate * grads["dw"]
        nb = nw - learning_rate * grads["db"]
        # cost[i] = cost
    params = {"w": nw, "b": nb}
    grads = {"dw": nw, "b": nb}
    return params, grads, cost
