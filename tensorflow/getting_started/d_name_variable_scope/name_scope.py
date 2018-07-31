import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# tf.name_scope 主要结合 tf.Variable() 来使用，方便参数命名管理。

'''
Signature: tf.name_scope(*args, **kwds)
Docstring:
Returns a context manager for use when defining a Python op.
'''
# 也就是说，它的主要目的是为了更加方便地管理参数命名。
# 与 tf.Variable() 结合使用。简化了命名
with tf.name_scope('conv1'):
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

# 下面是在另外一个命名空间来定义变量的
with tf.name_scope('conv2'):
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

# 所以，实际上weights1 和 weights2 这两个引用名指向了不同的空间，不会冲突
print(weights1.name)
print(weights2.name)
# 注意，这里的 with 和 python 中其他的 with 是不一样的
# 执行完 with 里边的语句之后，这个 conv1/ 和 conv2/ 空间还是在内存中的。这时候如果再次执行上面的代码
# 就会再生成其他命名空间
with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

print(weights1.name)
print(weights2.name)
