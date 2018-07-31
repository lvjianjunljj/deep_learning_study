import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# 下面是定义一个卷积层的通用方式
def conv_relu(kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    return None


def my_image_filter():
    # 按照下面的方式定义卷积层，非常直观，而且富有层次感
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu([5, 5, 32, 32], [32])


with tf.variable_scope("image_filters") as scope:
    # 下面我们两次调用 my_image_filter 函数，但是由于引入了 变量共享机制
    # 可以看到我们只是创建了一遍网络结构。
    result1 = my_image_filter()
    scope.reuse_variables()
    result2 = my_image_filter()

# 看看下面，完美地实现了变量共享！！！
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)
