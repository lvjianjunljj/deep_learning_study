import tensorflow as tf

state = tf.Variable(0, name = "Counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(10):
        print(sess.run(update))
