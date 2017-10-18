# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:55:48 2017

@author: n9544411
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:07:57 2017

@author: DoyoBae
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:02:07 2017
@author: doyoo

Title: Siamese network  based on a convolutional neural network algorithm with a multi-layer perceptron. 

"""

'''
1. Load the necessary libraries and start a tensorflow session
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.contrib.keras import backend as K

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import assign2_utils
import my_submission as my

sess = tf.Session()


def euclidean_distance(vects):
    '''
    Auxiliary function to compute the Euclidian distance between two vectors
    in a Keras layer.
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
'''
2. load the datasets

'''
'''
import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()
(x_train, _), (x_test, _) = data

'''
'''
#from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
path = 'C:/Users/doyobae/Downloads/mnist.npz' 
f = np.load(path)
x_original_train = f['x_train']
y_train = f['y_train']
x_original_test = f['x_test']
y_test = f['y_test']
f.close()
'''

x_original_train, y_train, x_original_test, y_test = assign2_utils.load_dataset()



#x_original_train, y_train, x_original_test, y_test = assign2_utils.load_dataset()

'''
3. pre-processed dataset
im2 = assign2_utils.random_deform(im1, 45, 0.3)
im1: A pre-wraped image
im2: A wraped image
the maximum degree: 45 
strength: 0.3

'''


'''
3.1  Orignal datasets.

'''

#x_original_train, y_train, x_original_test, y_test = assign2_utils.load_dataset()

'''
3.2 larger transfomred datasets

'''
x_largeTransformed_train = np.array([assign2_utils.random_deform(x,45,0.3) for x in x_original_train])
x_largeTransformed_test = np.array([assign2_utils.random_deform(x,45,0.3) for x in x_original_test])

'''
3.3 smaller transformed datasets.

'''
#x_smallTransformed_train = np.array([assign2_utils.random_deform(x,30,0.2) for x in x_original_train])
#x_smallTransformed_test = np.array([assign2_utils.random_deform(x,30,0.2) for x in x_original_test])

'''
4. Reshape the datasets and nomalised them.
(the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 )
'''
x_train = x_original_train.reshape(60000, 784)
x_test = x_original_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 # normalized the entries between 0 and 1
x_test /= 255
input_dim = 784 # 28x28

#train_xdata = np.array([np.reshape(x, (28,28)) for x in x_train])
num_classes = 2
epochs = 20
'''
5. Create the positive and negative pairs
(the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 )
'''
digit_indicesTrain = [np.where(y_train == i)[0] for i in range(10)]
#print(digit_indicesTrain)
tr_pairs, tr_y = my.create_pairs(x_train, digit_indicesTrain)
#print(tr_pairs, tr_y)
digit_indicesTest = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = my.create_pairs(x_test, digit_indicesTest)



#Convert images into 28x28 (they are downloaded as 1x784)

#https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
#shap the train
image_width = 28
image_height = 28
batch_size = 100
#learning_rate = 0.005
#evaluation_size = 500

#target_size = max(train_labels) + 1
num_channels = 1 # greyscale = 1 channel
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2 # NxN window for 1st max pool layer
max_pool_size2 = 2 # NxN window for 2nd max pool layer
fully_connected_size1 = 100


input_shape = (image_width, image_height, num_channels)



def create_one_architecture(input_shape, num_classes):

    seq  = keras.models.Sequential()
    ##how to select the initial weight at tensorflow book
    ##how to decide max pooling2D?
    ##activation and dense and activation
    ##drop out
    #Keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
  
    # Output shape of convolution is 4d
    '''
    Keras' convention is that the batch dimension (number of examples (not the same as timesteps)) is typically omitted in the input_shape arguments. The batching (number of examples per batch) is handled in the fit call. I hope that helps. Thanks.
    '''
    '''
    output_size = 64
    kernel_size = (2,2)
    pool_size=(2, 2)
    model.add(keras.layers.Conv2D(output_size, kernel_size, activation='relu',input_shape= (input_shape), kernel_initializer='glorot_uniform'))
    #model.add(keras.layers.Conv2d( ,(26,),  strides=[1, 1, 1, 1],  activation='relu',input_shape=(10, input_dim), kernel_initializer='glorot_uniform'))

    model.add(keras.layers.MaxPooling2D(pool_size, strides=(2,2), padding='valid', data_format=None))
    
    model.add(keras.layers.Dropout(0.1))
    
    model.add(keras.layers.Flatten()) # Dense layer require a 2d input

    #model.add(keras.layers.Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    
    model.add(keras.layers.Dense(128, activation = 'relu')) 

    model.add(keras.layers.Dropout(0.1))
    
    model.add(keras.layers.Dense(10))
    #model.add(Dense(num_classes, activation='softmax'))
    '''

    seq.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    seq.add(keras.layers.Dropout(0.25))
    seq.add(keras.layers.Flatten())
    seq.add(keras.layers.Dense(128, activation='relu'))
    seq.add(keras.layers.Dropout(0.5))  
    seq.add(keras.layers.Dense(128, activation='relu'))
    seq.add(keras.layers.Dropout(0.5))


    return seq

#https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf


# Declare model placeholders

tr_pair1 = np.array([np.reshape(x, (image_width, image_height, num_channels)) for x in tr_pairs[:, 0]])
tr_pair1 = tr_pair1.reshape(tr_pair1.shape[0],tr_pair1.shape[1],tr_pair1.shape[2], num_channels)
tr_pair2  = np.array([np.reshape(x, (image_width, image_height), num_channels) for x in tr_pairs[:, 1]])
tr_pair2 = tr_pair2.reshape(tr_pair2.shape[0],tr_pair2.shape[1],tr_pair2.shape[2], num_channels)

te_pair1 = np.array([np.reshape(x, (image_width,image_height), num_channels) for x in te_pairs[:, 0]])
te_pair1 = te_pair1.reshape(te_pair1.shape[0],te_pair1.shape[1],te_pair1.shape[2], num_channels)

te_pair2= np.array([np.reshape(x, (image_width,image_height), num_channels) for x in te_pairs[:, 1]])
te_pair2 = te_pair2.reshape(te_pair2.shape[0],te_pair2.shape[1],te_pair2.shape[2], num_channels)

'''

tr_pair1 = tr_pairs[:, 0].reshape( tr_pairs[:, 0].shape[0], (image_width,image_height), num_channels) 
tr_pair2  = tr_pairs[:, 1].reshape( tr_pairs[:, 1].shape[0], (image_width,image_height), num_channels) 


te_pair1 = te_pairs[:, 0].reshape( te_pairs[:, 0].shape[0], (image_width,image_height), num_channels) 
te_pair2  = te_pairs[:, 1].reshape( te_pairs[:, 1].shape[0], (image_width,image_height), num_channels) 
'''
'''
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))



'''


print(tr_pairs[:, 0])
# network definition
base_network = create_one_architecture(input_shape,num_classes)

#two twin network
input_a = keras.layers.Input(shape=(input_shape))
input_b = keras.layers.Input(shape=(input_shape))


#np_utils.to_categorical
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches


processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])

    
# node to compute the distance between the two vectors
# processed_a and processed_a
#distance = keras.layers.Lambda(my.euclidean_distance)([processed_a, processed_b])
# Our model take as input a pair of images input_a and input_b
# and output the Euclidian distance of the mapped inputs

model = keras.models.Model([input_a, input_b], distance)


epochs = 2
batch_size=128

rms = keras.optimizers.RMSprop()
model.compile(loss=my.contrastive_loss, optimizer=rms)#sgd
model.fit([tr_pair1, tr_pair2], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([te_pair1, te_pair2], te_y))

pred = model.predict([tr_pair1, tr_pair2])
tr_acc = my.compute_accuracy(pred, tr_y)
pred = model.predict([te_pair1, te_pair2])
te_acc = my.compute_accuracy(pred, te_y)



print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))



'''



print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))




for i in range(300):
    l = model.train_on_batch(pairs_train, y_train)[0]
    pred_train = model.predict(pairs_train)
    pred_test = model.predict(pairs_test)
    print('loss=%0.6f accuracy (train)=%0.2f%% accuracy (test)=%0.2f%%' %
          (l, 100*compute_accuracy(pred_train), 100*compute_accuracy(pred_test)))
# 
train_labels = mnist.train.labels
test_labels = mnist.test.labels
'''

'''
if K.image_data_format() == 'channels_first':
    



    x_train = tr_pairs[:, 0].reshape(x_train.shape[0], num_channels ,image_width, image_height)
    x_test = tr_pairs[:, 1].reshape(x_test.shape[0], num_channels, image_width, image_height)
    y_train = te_pairs[:, 0].reshape(x_train.shape[0], image_width, image_height, num_channels)
    y_test = tr_pairs[:, 1].reshape(x_test.shape[0], image_width, image_height, num_channels)
    input_shape = (num_channels, image_width, image_height)
else:
'''

'''
tr_pair1 = tr_pairs[:, 0].reshape(x_train.shape[0], image_width, image_height , num_channels)
tr_pair2 = tr_pairs[:, 1].reshape(x_test.shape[0], num_channels, image_width, image_height)
te_pair1 = te_pairs[:, 0].reshape(x_train.shape[0], image_width, image_height, num_channels)
te_pair2 = tr_pairs[:, 1].reshape(x_test.shape[0], image_width, image_height, num_channels)
'''
    #abbrevate this one
'''
train_pair1 = np.array([np.reshape(x, (28,28)) for x in te_pairs[:, 0]])
train_pair2 = np.array([np.reshape(x, (28,28)) for x in te_pairs[:, 1]])

test_pair1 = np.array([np.reshape(x, (28,28)) for x in te_pairs[:, 0]])
test_pair2 = np.array([np.reshape(x, (28,28)) for x in te_pairs[:, 1]])
'''
'''
    chage teh input shape according to tr_pair size
    kernel_size = (2,2),
    model.add(Convolution2D(...)) 
    model.add(Flatten()) # Flatten input into 2d
    model.add(Dense(...)) # Dense layer require a 2d input
    kernel_regularizer=l2(2e-4)
    model.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128,(4,4),activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(2e-4),bias_initializer='zeros'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256,(4,4),activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(2e-4),bias_initializer='zeros'))
    model.add(Flatten())
    '''
'''
    model.add(Activation('relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(Dense(128))
    #model.add(Activation('relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(Dense(10))
    #model.add(Activation('softmax'))
'''