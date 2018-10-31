import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session()

# 1.placeholder
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4])
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
print(v1.name)
print(type(v1))
print(v1)
print("......................................................")

# 2. tf.Variable()
v2 = tf.Variable([1, 2], dtype=tf.float32)
print(v2.name)
v2 = tf.Variable([1, 2], dtype=tf.float32, name='V')
print(v2.name)
v2 = tf.Variable([1, 2], dtype=tf.float32, name='V')
print(v2.name)
print(type(v2))
print(v2)
print("......................................................")

# 3.tf.get_variable() 创建变量的时候必须要提供 name
v3 = tf.get_variable(name='gv', shape=[])
print(v3.name)
v4 = tf.get_variable(name='gv', shape=[2])
print(v4.name)
print("......................................................")
