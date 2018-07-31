from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# 我们先来看看数据是什么样的
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
img1 = mnist.train.images[1]
label1 = mnist.train.labels[1]
print(label1)  # 所以这个是数字 6 的图片
print('img_data shape =', img1.shape)  # 我们需要把它转为 28 * 28 的矩阵
img1.shape = [28, 28]

# import matplotlib.image as mpimg  # 用于读取图片，这里用不上

print(img1.shape)

# 我们可以通过设置 cmap 参数来显示灰度图
plt.imshow(img1, cmap='gray') # 'hot' 是热度图
plt.axis('off') # 不显示坐标轴
plt.show()

#我们想看 Conv1 层的32个卷积滤波后的结果，显示在同一张图上。 python 中也有 plt.subplot(121) 这样的方法来帮我们解决这个问题。
plt.subplot(4,8,1)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.subplot(4,8,2)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.show()