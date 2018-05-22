import tensorflow as tf

tf.enable_eager_execution()


res = [[1,2], [3,4]]
print res
output = tf.cumsum(res)
print output
# a = [1, -2, 0, 3, -0.1, float('NaN')]
# m = tf.nn.relu(a)
# print m

