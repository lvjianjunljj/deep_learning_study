import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph("D:\\DeepLearning\\digitalRecognition\\modelGallery\\autoGraphTF\\MyModel.ckpt-1000.meta")
    saver.restore(sess, tf.train.latest_checkpoint('D:\\DeepLearning\\digitalRecognition\\modelGallery\\autoGraphTF\\'))
    print(sess.run('w1:0'))
