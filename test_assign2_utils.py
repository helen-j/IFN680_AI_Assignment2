

'''

A short script to illustrate the warping functions of 'assign2_utils'

'''

#import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib import keras
#from tensorflow.contrib.keras import backend as K


import assign2_utils



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

im1 = x_train[20]

plt.imshow(im1,cmap='gray')


im2 = assign2_utils.random_deform(im1,45,0.3)

plt.figure()

plt.imshow(im2,cmap='gray')

plt.show()