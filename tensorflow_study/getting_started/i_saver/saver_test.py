import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create some variables.
v1 = tf.Variable([1.0, 2.3], name="v1")
v2 = tf.Variable(55.5, name="v2")

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

ckpt_path = './ckpt/test-model.ckpt'
# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
sess.run(init_op)
save_path = saver.save(sess, ckpt_path, global_step=1)
print("Model saved in file: %s" % save_path)