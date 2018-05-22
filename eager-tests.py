import tensorflow as tf

tf.enable_eager_execution()

# NaN:
# a = [1, -2, 0, 3, -0.1, float('NaN')]
# m = tf.nn.relu(a)
# print m

# NotEqual:
# a = tf.constant([[[1], [4], [5]], [[8], [9], [12]]], shape=[2, 3, 1], dtype=tf.int32)
# b = tf.constant([[[2], [3], [6]], [[7], [10], [12]]], shape=[2, 3, 1], dtype=tf.int32)
# c = tf.not_equal(a, b)
# print c

# a = tf.constant([1, 2, -3, 5], shape=[2, 2])
# dy = tf.constant([1, 2, 3, 4], shape=[2, 2])

a = tf.constant(0.)
b = 2 * a
g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
print g

# SoftMax:
# logits = tf.constant([1, 2, 3], dtype=tf.float32)
# labels = tf.constant([0.3, 0.6, 0.1])

# softmaxLogits = tf.nn.softmax(logits)
# print softmaxLogits

# y = tf.losses.softmax_cross_entropy(labels, logits)
# print y

# print softmaxLogits[0]
# print softmaxLogits[1]
# print softmaxLogits[2]
