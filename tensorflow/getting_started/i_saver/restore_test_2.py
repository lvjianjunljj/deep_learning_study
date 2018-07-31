import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create some variables.
v1 = tf.Variable([11.0, 16.3], name="v1")


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# Restore variables from disk.
ckpt_path = './ckpt/test-model.ckpt'
saver.restore(sess, ckpt_path + '-' + str(1))
print("Model restored.")

print(sess.run(v1))
