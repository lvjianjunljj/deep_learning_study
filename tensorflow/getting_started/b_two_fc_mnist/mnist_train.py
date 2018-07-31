import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
# 用tensorflow 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print('training data shape ', mnist.train.images.shape)
print('training label shape ', mnist.train.labels.shape)


# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# input_layer
X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# FC1
W_fc1 = weight_variable([784, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(X_, W_fc1) + b_fc1)
# FC2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pre = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# 1.损失函数：cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pre))
# 2.优化函数：AdamOptimizer, 优化速度要比 GradientOptimizer 快很多
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 3.预测结果评估
# 　预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 开始运行
sess.run(tf.global_variables_initializer())
# 这大概迭代了不到 10 个 epoch， 训练准确率已经达到了0.98
for i in range(1000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)
    train_step.run(feed_dict={X_: X_batch, y_: y_batch})
    if (i + 1) % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={X_: mnist.train.images, y_: mnist.train.labels})
        print("step %d, training acc %g" % (i + 1, train_accuracy))
    if (i + 1) % 1000 == 0:
        test_accuracy = accuracy.eval(feed_dict={X_: mnist.test.images, y_: mnist.test.labels})
        print("= " * 10, "step %d, testing acc %g" % (i + 1, test_accuracy))
saver = tf.train.Saver(max_to_keep=1)
saver.save(sess, "D:/DeepLearning/digitalRecognition/modelGallery/getting_started/TwoFC.ckpt", global_step=1000)
