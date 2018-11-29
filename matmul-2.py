import tensorflow as tf
import numpy as np

tf.enable_eager_execution()
a = np.array([[2, 3, 6], [4, 4, 7]], dtype=np.int32)
b = np.array([[1, 2], [4, 23], [1, 3]], dtype=np.int32)
print(a)

A = tf.expand_dims(a, 2)
B = tf.expand_dims(b, 0)

print(A)

C = tf.reduce_sum(A * B, 1)

print(C)
