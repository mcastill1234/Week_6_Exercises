import numpy as np

# Exercise 1: Prediction
print("Exercise 1: Prediction:\n")

# Exercise 1A)
X = np.array([[0, 1, 2], [0, 1, 2]])
Y = np.array([[0, 1, 0]])
W1 = np.array([[1, -1], [0, 0]])
W1_0 = np.array([[-0.5], [1.5]])

Z1 = W1.T @ X + W1_0
FZ1 = np.heaviside(Z1, 1)

print("1A: First layer outputs: ", FZ1.tolist())

# Exercise 1B)
# We pick W2 = [1, 1].T and W2_0 = [-1.5]
W2 = np.array([1, 1])
W2_0 = np.array([-1.5])
Z2 = W2.T @ FZ1 + W2_0
FZ2 = np.heaviside(Z2, 1)
print("1B: Second layer outputs: ", FZ2.tolist())
print("    Original Labels: ", Y.tolist())
print()

# Exercise 2: Training
print("Exercise 2: Training:\n")

# Exercise 2A)
W = np.array([[1], [1]])
W_0 = np.array([[1]])
x = np.array([[1, 2]]).T
y = np.array([-1])
a11 = W.T @ x + W_0
print("2A: The value of a11 given input x_i = [1, 2].T is: ", a11.tolist())


# Exercise 2B)
W1 = np.zeros((2, 1))
W1[0] = W[0] - 0.5 * -y[0] * x[0, 0]
W1[1] = W[1] - 0.5 * -y[0] * x[1, 0]
W1_0 = W_0[0] - 0.5 * -y[0]
print("2B: The value of the weights after one step of SGD are: W0, [W] =", W1_0, W1.tolist())

# Exercise 2C)
a11 = W1.T @ x + W1_0
print("2C: The value of a11 with these new weights is", a11.tolist())

# Exercise 2D)
W2 = np.zeros((2, 1))
W2[0] = W1[0] - 0.5 * -y[0] * x[0, 0]
W2[1] = W1[1] - 0.5 * -y[0] * x[1, 0]
W2_0 = W1_0[0] - 0.5 * -y[0]
print("2D: The value of the weights after one more step of SGD are: W0, [W] =", W2_0, W2.tolist())

# Exercise 2E)
a11 = W2.T @ x + W2_0
print("2D: The value of a11 with these new weights is", a11.tolist())

# Exercise 2F)
print("2F: If we do one more SDG update, nothing happens; the gradients are zero since the hinge loss is zero")