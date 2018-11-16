import random
import numpy as np
from keras.utils import np_utils

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [i for j in a for i in j]
print(a)
print(b)
