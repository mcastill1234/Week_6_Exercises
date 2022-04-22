import numpy as np

# Exercise 1A:
X = np.array([[0, 1, 2], [0, 1, 2]])
Y = np.array([[0, 1, 0]])
W = np.array([[1, -1], [0, 0]])
W0 = np.array([[-0.5], [1.5]])

Z = W.T @ X + W0
FZ = np.heaviside(Z, 1)

print("First layer outputs", FZ.tolist())





