import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()

a = np.array([2, 4, 1])
A = tf.expand_dims(a, 1)
print(A)
# print(np.zeros(shape=[np.size(a, 0), tf.reduce_max(a), 1]))
B = tf.range(0, tf.reduce_max(a), 1) + np.zeros(shape=[np.size(a, 0), tf.reduce_max(a)], dtype=np.int32)
C = B < A
tf.reshape(B, shape=[np.size(a) * tf.reduce_max(a)])
tf.reshape(C, shape=[np.size(a) * tf.reduce_max(a)])

Ans = tf.boolean_mask(B, C)

print(Ans)
