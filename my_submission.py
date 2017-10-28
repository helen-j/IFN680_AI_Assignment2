'''

2017 IFN680 Assignment Two

Doyoo Baek N9544411
Helen Jeffrey N9416528

'''


import random
import numpy as np
import tensorflow as tf

from tensorflow.contrib import keras
from tensorflow.contrib.keras import backend as K

import assign2_utils

sess = tf.Session()

#------------------------------------------------------------------------------
# Define parameters:
#------------------------------------------------------------------------------
# image dimensions    
image_width     = 28
image_height    = 28
num_channels    = 1         # greyscale = 1 channel
#input_shape     = 

# warp parameters
degree          = 0         # used to warp the original data set
strength        = 0         # used to warp the original data set

# traing/validation
training_epochs = 1
train_size      = 60000     # Training size
val_size        = 40000     # Validation size

# Experiment Architecture
architecture_flag    = 1    # which network architecture to use
                            # 1 = simplistic, 2 = CNN_2, 3 = CNN_3
                            
#------------------------------------------------------------------------------

def euclidean_distance(vects):
    '''
    Auxiliary function to compute the Euclidian distance between two vectors in a Keras layer.    
    @param
        vects: the two vectors to be compared
    @return
        euclidean_distance between two vectors    
    '''

    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#------------------------------------------------------------------------------

def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
      y_true : true label 1 for positive pair, 0 for negative pair
      y_pred : distance output of the Siamese network 
    @return
        contrastive_loss between two values   
    '''

    margin = 1

    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network
    # if negative pair, y_true is 0, penalize for distance smaller than the margin

    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

#------------------------------------------------------------------------------


def compute_accuracy(predictions, labels):
    '''
    Compute classification true positive accuracy with a fixed threshold on distances.
    @param 
      predictions : values computed by the Siamese network
      labels : 1 for positive pair, 0 otherwise
    @return
        return labels[predictions.ravel() < 0.5].mean()
    '''    
    
    n = labels.shape[0]
    acc =  (labels[predictions.ravel() < 0.5].sum() +  # count True Positive
               (1-labels[predictions.ravel() >= 0.5]).sum() ) / n  # True Negative
    return acc


#------------------------------------------------------------------------------

def create_pairs(x, digit_indices):
    '''
       Positive and negative pair creation.
       Alternates between positive and negative pairs.
       @param
         digit_indices : list of lists
            digit_indices[k] is the list of indices of occurences digit k in 
            the dataset
       @return
         P, L 
         where P is an array of pairs and L an array of labels
         L[i] ==1 if P[i] is a positive pair
         L[i] ==0 if P[i] is a negative pair
         
    '''
    
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    if(n>5000):
        n = 5000
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # z1 and z2 form a positive pair
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # z1 and z2 form a negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
   


#------------------------------------------------------------------------------
def compute_performance():
    
    
    pass



#------------------------------------------------------------------------------
def train_siamese(model, tr_pairs, tr_y, te_pairs, te_y):
    
        # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=training_epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    
#    pass



#------------------------------------------------------------------------------        

def transform_dataset(train_dataset, test_dataset, degree, strength):
    
    '''       
        Data preprocessing:
        Transform the original data set

       @param
        dataset: The source dataset that will be transformed
        degree: The max degree of the transformation
        strength: The strength of the transformation

       @return
        x_Transformed_train: 
        x_Transformed_test: 

    '''

    x_Transformed_train = train_dataset + np.array([assign2_utils.random_deform(x,degree,strength) for x in train_dataset])
        
    
    x_Transformed_test = np.array([assign2_utils.random_deform(x,degree,strength) for x in test_dataset])
    return x_Transformed_train, x_Transformed_test    


#------------------------------------------------------------------------------
def reshape_input(image_width, image_height, num_channels, pairs):
    
    '''       
    This function will reshape input for 2D Convolution:

       @param
        image_width, image_height, num_channels : defined at top of code
        pairs : 

       @return
        Twin1_input_pair: 
        Twin2_input_pair: 

    '''    
    
    #reshape the image as 28*28 
    Twin1_input_pair = np.array([np.reshape(x, (image_width, image_height)) for x in pairs[:, 0]])
    #shape the input for twin1 neural network
    Twin1_input_pair = Twin1_input_pair.reshape(Twin1_input_pair.shape[0],Twin1_input_pair.shape[1],pairs.shape[2], num_channels)
    #reshape the image as 28*28  
    Twin2_input_pair = np.array([np.reshape(x, (image_width, image_height)) for x in pairs[:, 1]])
    #shape the input for twin2 neural network
    Twin2_input_pair = Twin2_input_pair.reshape(Twin2_input_pair.shape[0],Twin2_input_pair.shape[1],pairs.shape[2], num_channels)
   
    return  Twin1_input_pair,  Twin2_input_pair


#------------------------------------------------------------------------------
def preprocess_Data(degree, strength):
    '''
    This function prepares the data set for the training stage (and validation)
    1. Load the data
    2. Transform data (if warping is needed)
    3. Create the positive and negative pairs
    4. Reshape the pairs for 2D convolutional layer
    5. Split into training / validation datasets   
    
       @param
        degree: The max degree of the transformation
        strength: The strength of the transformation
        percentage : Percentage of dataset used for training
       @return
      :
    '''    
    
    #   1. load the data
    x_original_train, train_target, x_original_test, val_target = assign2_utils.load_dataset()
    #   2. transformed data
    if (degree != 0 and strength !=0 ):
        train_pairs, val_pairs = transform_dataset(x_original_train, x_original_test, degree, strength)
    else:
        train_pairs, val_pairs = x_original_train, x_original_test


    #   3. Create the positive and negative pairs
    #   the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 

    digit_indicesTrain = [np.where(train_target == i)[0] for i in range(10)]
    train_pairs, train_y = create_pairs(train_pairs, digit_indicesTrain)
    
    #
    digit_indicesTest = [np.where(val_target == i)[0] for i in range(10)]
    val_pairs, val_y = create_pairs(val_pairs, digit_indicesTest)

    #   4. reshape the pairs for 2D  convolutional layer
    train_twin1_input, train_twin2_input = reshape_input(image_width, image_height, num_channels, train_pairs)
    val_twin1_input, val_twin2_input = reshape_input(image_width, image_height, num_channels, train_pairs)
    return train_twin1_input, train_twin2_input, val_twin1_input, val_twin2_input


#------------------------------------------------------------------------------

def solution(architecture_flag, degree, strength, percentage):
    
#    def baseline_Architecture(input_shape): 
#        seq = keras.models.Sequential()
#        seq.add(keras.layers.Dense(128, input_shape=(input_dim,), activation='relu'))
#        seq.add(keras.layers.Dropout(0.1))
#        seq.add(keras.layers.Dense(128, activation='relu'))
#        seq.add(keras.layers.Dropout(0.1))
#        seq.add(keras.layers.Dense(128, activation='relu'))
#        return seq
    
    
    def first_Architecture(input_shape): 
        seq  = keras.models.Sequential()
        #Keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        # Output shape of convolution is 4d
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
    
    def second_Architecture(input_shape): 
        seq  = keras.models.Sequential()
        #Keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        # Output shape of convolution is 4d
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
    
    input_shape = (image_width, image_height, num_channels)
    
    train_twin1_input, train_twin2_input, val_twin1_input, val_twin2_input = preprocess_Data(degree, strength)
    
    if(architecture_flag == 1): 
        base_network = first_Architecture(input_shape)
    else: 
        base_network = second_Architecture(input_shape)
    input_a = keras.layers.Input(shape=(input_shape))
    input_b = keras.layers.Input(shape=(input_shape))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # node to compute the distance between the two vectors
    # processed_a and processed_a
    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])
    
    # Our model take as input a pair of images input_a and input_b
    # and output the Euclidian distance of the mapped inputs
    model = keras.models.Model([input_a, input_b], distance)
    
#    train_siamese(model, tr_pairs, tr_y, te_pairs, te_y)
    

    '''
    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    '''

#------------------------------------------------------------------------------

if __name__=='__main__':
    solution(architecture_flag = 1, degree = 45, strength = 0.3, percentage = 60)


    



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#                               CODE CEMETARY        


'''
def BatchNormalisation_architecture(input_shape, num_classes):
    seq  = keras.models.Sequential()
    ##how to select the initial weight at tensorflow book
    ##how to decide max pooling2D?
    ##activation and dense and activation
    ##drop out
    #Keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
  
    # Output shape of convolution is 4d

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


def BatchNormalisation_architecture(input_shape, num_classes):
    seq  = keras.models.Sequential()
    ##how to select the initial weight at tensorflow book
    ##how to decide max pooling2D?
    ##activation and dense and activation
    ##drop out
    #Keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
  
    # Output shape of convolution is 4d

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
'''