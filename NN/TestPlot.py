import matplotlib.pyplot as plt
import numpy as np

testcost = np.random.random((1, 20)) * 100
testcost = np.squeeze(testcost)
testcost1 = np.random.random((1, 10)) * 100
testcost1 = np.squeeze(testcost1)
plt.plot(testcost, label="123")
plt.plot(testcost1, label="333")
plt.title("test")
plt.xlabel("iteration")
plt.ylabel("cost")
legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")

plt.show()
