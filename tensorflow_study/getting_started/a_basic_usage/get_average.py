import tensorflow as tf

h_sum = tf.Variable(0.0, dtype=tf.float32)
# h_vec = tf.random_normal(shape=([10]))
h_vec = tf.constant([1.0, 2.0, 3.0, 4.0])
# 把 h_vec 的每个元素加到 h_sum 中，然后再除以 10 来计算平均值
# 待添加的数
h_add = tf.placeholder(tf.float32)
# 添加之后的值
h_new = tf.add(h_sum, h_add)
# 更新 h_new 的 op
update = tf.assign(h_sum, h_new)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 查看原始值
    print
    's_sum =', sess.run(h_sum)
    print
    "vec = ", sess.run(h_vec)
    # 循环添加
    for i in range(4):

        sess.run(update, feed_dict={h_add: sess.run(h_vec[i])})
        print('h_sum =', sess.run(h_sum))
    #     print 'the mean is ', sess.run(sess.run(h_sum) / 4)  # 这样写 4  是错误的， 必须转为 tf 变量或者常量
    print('the mean is ', sess.run(sess.run(h_sum) / tf.constant(4.0)))
