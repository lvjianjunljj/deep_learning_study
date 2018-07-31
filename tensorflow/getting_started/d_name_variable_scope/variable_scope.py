# 这里是正确的打开方式~~~可以看出，name 参数才是对象的唯一标识
import tensorflow as tf

with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2, 3])
    bias1 = tf.get_variable('bias', shape=[3])

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')

print(Weights1.name)
print(Weights2.name)
# 可以看到这两个引用名称指向的是同一个内存对象


# 注意， bias1 的定义方式
with tf.variable_scope('v_scope', reuse=True):
    Weights1 = tf.get_variable('Weights', shape=[2, 3])
    bias1 = tf.Variable([0.52], name='bias')

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True):
    Weights2 = tf.get_variable('Weights')
    bias2 = tf.Variable([0.52], name='bias')

print(Weights1.name)
print(Weights2.name)
print(bias1.name)
print(bias2.name)
