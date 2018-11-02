# The most successful model save demo!!!
# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
import tensorflow as tf
import os
# define input variables
x = tf.placeholder(tf.float32, [None, 784])
# define parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# define activation function
y = tf.nn.softmax(tf.matmul(x, W) + b)
# define output variables
y_ = tf.placeholder(tf.float32, [None, 10])
# define loss functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# define optimization function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化变量
init = tf.global_variables_initializer()
# 定义会话
sess = tf.Session()
# 定义模型保存对象
saver = tf.train.Saver()
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)
# 运行初始化
sess.run(init)
# 循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
print("train complete!!!")
# 创建模型保存目录
model_dir = r"D:/DeepLearning/digitalRecognition/modelGallery/ModelSaveRestore2"
model_name = "MyModel.ckpt"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# save model
saver.save(sess, os.path.join(model_dir, model_name))
print("save model success!!!")