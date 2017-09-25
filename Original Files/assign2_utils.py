'''

Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''

import numpy as np

import random

from tensorflow.contrib import keras


from tensorflow.contrib.keras import backend as K

from skimage import transform

#------------------------------------------------------------------------------

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#------------------------------------------------------------------------------

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

#------------------------------------------------------------------------------

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

#------------------------------------------------------------------------------

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


#------------------------------------------------------------------------------

def load_dataset():
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test
    The dtype of all returned array is uint8
    
    @return
        x_train, y_train, x_test, y_test
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    return x_train, y_train, x_test, y_test

#------------------------------------------------------------------------------
    
def random_homography(variation, image_side):
    '''
    Generate a random homography.
    
    The homography is defined by 4 random points.

       @param
       
           variation:    percentage (in decimal notation from 0 to 1)
                         relative size of a circle region where centre is projected
                         
           image_side:   
                         length of the side of an input square image in pixels
       
       @return
       
           tform:        object from skimage.transfrm
    
    '''
    d = image_side * variation
    
    top_left =    (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))  # Top left corner
    bottom_left = (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))   # Bottom left corner
    top_right =   (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))     # Top right corner
    bottom_right =(random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))  # Bottom right corner

    tform = transform.ProjectiveTransform()
    tform.estimate(np.array((
            top_left,
            (bottom_left[0], image_side - bottom_left[1]),
            (image_side - bottom_right[0], image_side - bottom_right[1]),
            (image_side - top_right[0], top_right[1])
        )), np.array((
            (0, 0),
            (0, image_side),
            (image_side, image_side),
            (image_side, 0)
        )))       

    return tform

#------------------------------------------------------------------------------
  
    
def random_deform(image, rotation, variation, image_side=28):

        cval = 0
        rhom = random_homography(variation, image_side)
        image_warped = transform.rotate(
            image, 
            random.uniform(-rotation, rotation), 
            resize = False,
            mode='constant', 
            cval=cval)
        image_warped = transform.warp(image_warped, rhom, mode='constant', cval=cval)
        return image_warped

  
#------------------------------------------------------------------------------
    
    
