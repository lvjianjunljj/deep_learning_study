import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 拿官方的例子改动一下
def my_image_filter():
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    return None

# First call creates one set of 4 variables.
result1 = my_image_filter()
# Another set of 4 variables is created in the second call.
result2 = my_image_filter()
# 获取所有的可训练变量
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)