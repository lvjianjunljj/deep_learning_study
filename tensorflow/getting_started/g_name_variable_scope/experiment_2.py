import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.name_scope('nsc1'):
    v1 = tf.Variable([1], name='v1')
    with tf.variable_scope('vsc1'):
        v2 = tf.Variable([1], name='v2')
        v3 = tf.get_variable(name='v3', shape=[])
print('v1.name: ', v1.name)
print('v2.name: ', v2.name)
print('v3.name: ', v3.name)
print("......................................................")

with tf.name_scope('nsc1'):
    v4 = tf.Variable([1], name='v4')
print('v4.name: ', v4.name)
