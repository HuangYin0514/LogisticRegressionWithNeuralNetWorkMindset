# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from lr_utils import load_dataset

'''
Define the model structure (such as number of input features)
Initialize the model's parameters
Loop:
    Calculate current loss (forward propagation)
    Calculate current gradient (backward propagation)
    Update parameters (gradient descent)
You often build 1-3 separately and integrate them into one function we call model().
'''

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
# plt.show()
print("y = " + str(train_set_y[:, index]) + ", it is a " +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + " picture")
print()
### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

### END CODE HERE ###
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print()

# Reshape the training and test examples
### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###
print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

# broadcast
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# sigma functions
from Sigmoid import sigmoid

print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(9.2) = " + str(sigmoid(9.2)))
print()

# Initializing parameters
from Initialize_with_zeros import initialize_with_zeros

dim = 2
w, b = initialize_with_zeros(2)
print("w = " + str(w))
print("b = " + str(b))
print()

# propagate
from Propagate import propagate

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))
print()

# optimize
from Optimize import optimize

params, grads, costs = optimize(w, b, X, Y, num_iterations=101, learning_rate=0.005, print_cost=True)
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(costs))

# pridict
from Predict import predict

print("prediction = " + str(predict(w, b, X)))
print()

# model
from Model import model

d = model(X_train=train_set_x, Y_train=train_set_y, X_test=test_set_x, Y_test=test_set_y, num_iteration=2000,
          learn_rate=0.005, print_cost=True)
print()

# Example of a picture that was wrongly classified.
index = 5
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
# plt.show()
print("y = " + str(test_set_y[:, index]) +
      ", you pridict that it is a \"" + classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture")

# plot the cost function and the gradient
# Plot learning curve (with costs)
costs = np.squeeze(d["costs"])
plt.figure()
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per handreds)")
plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.show()
print()

# further analysis(optionanl/ungraded exersicse)
# choice of learning rate

learing_rate = [0.01, 0.001, 0.0001]
models = {}
plt.figure()
for i in learing_rate:
    print("learing rate is " + str(i))
    models[str(i)] = model(X_train=train_set_x, Y_train=train_set_y, X_test=test_set_x, Y_test=test_set_y,
                           num_iteration=15,
                           learn_rate=i, print_cost=False)
    print("\n" + "--------------------------------------------------" + "\n")

for i in learing_rate:
    plt.plot(models[str(i)]["costs"], label=models[str(i)]["learning_rate"])
plt.xlabel("iterations")
plt.ylabel("cost")
legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
# plt.show()

my_image = "timg.jpg"
# we Preprocess the image to fit your algorithm
fname = "images/" + my_image
image = np.array(plt.imread(fname))
my_image = scipy.resize(image, (num_px, num_px,3)).reshape((1,num_px*num_px*3)).T
my_predicted_image = predict(d["w"],d["b"],my_image)
plt.imshow(image)
# plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

print()

