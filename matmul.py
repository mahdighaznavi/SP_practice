import tensorflow as tf
import numpy as np

A = np.array([[2, 3, 6], [4, 4, 7]])
B = np.array([[1, 2], [4, 23], [1, 3]])

A = np.expand_dims(A, 2)
B = np.expand_dims(B, 0)

C = np.sum(A*B, 1)

print(C)
