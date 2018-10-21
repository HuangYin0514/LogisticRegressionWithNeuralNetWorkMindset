import numpy as np


def model(X_train, Y_train, X_test, Y_test, num_iteration=2000, learn_rate=0.5, print_cost=False):
    """
       Builds the logistic regression model by calling the function you've implemented previously

       Arguments:
       X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
       Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
       X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
       Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
       num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
       learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
       print_cost -- Set to true to print the cost every 100 iterations

       Returns:
       d -- dictionary containing information about the model.
       """

    # initialize parameters with zeros (≈ 1 line of code)
    from Initialize_with_zeros import initialize_with_zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    from Optimize import optimize
    # Gradient descent
    params, grads, costs = optimize(w, b, X_train, Y_train,
                                    num_iterations=num_iteration, learning_rate=learn_rate,
                                    print_cost=print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = params["w"]
    b = params["b"]

    # Predict test/train set examples
    from Predict import predict
    Y_pridiction_train = predict(w=w, b=b, X=X_train)
    Y_pridiction_test = predict(w=w, b=b, X=X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pridiction_train - Y_train)) * 100))
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pridiction_test - Y_test)) * 100))

    d = {
        "cost": costs,
        "w": w,
        "b": b,
        "Y_prediction_train": Y_pridiction_train,
        "Y_prediction_test": Y_pridiction_test,
        "learn_rate": learn_rate,
        "num_iterations": num_iteration
    }
    return 0
