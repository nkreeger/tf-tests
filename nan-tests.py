import tensorflow as tf

def cfg():
  optimizer_options = tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L0,
        do_constant_folding=False)
  graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
  return tf.ConfigProto(log_device_placement=True, graph_options=graph_options)

with tf.Graph().as_default():
    a = tf.placeholder(tf.float32, shape=(6,))
    b = tf.nn.relu(a)

    with tf.Session(config=cfg()) as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(b, feed_dict={a: [1, -2, 0, 3, -0.1, float('NaN')]}))
