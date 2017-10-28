'''



2017 IFN680 Assignment Two



Scaffholding code to get you started for the 2nd assignment.





'''

import random

import numpy as np

import tensorflow as tf

from tensorflow.contrib import keras
from tensorflow.contrib.keras import backend as K

import assign2_utils





sess = tf.Session()


#------------------------------------------------------------------------------




def euclidean_distance(vects):

    '''

    Auxiliary function to compute the Euclidian distance between two vectors

    in a Keras layer.

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

    '''

    margin = 1

    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network

    # if negative pair, y_true is 0, penalize for distance smaller than the margin

    return K.mean(y_true * K.square(y_pred) +

                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

#------------------------------------------------------------------------------



def compute_accuracy(predictions, labels):

    '''

    Compute classification accuracy with a fixed threshold on distances.

    @param 

      predictions : values computed by the Siamese network

      labels : 1 for positive pair, 0 otherwise

    '''
    # the formula below, compute only the true positive rate]

    #    return labels[predictions.ravel() < 0.5].mean()

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
    #to create the 100,000 training dataset control the number of n
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

    



#------------------------------------------------------------------------------        

def transform_dataset(train_dataset, test_dataset ,degree, strength):

    '''       
        Data preprocessing

        Transform the original data set

       @param
        dataset: The source dataset that will be transformed
        degree: The max degree of the transformation
        strength: The strength of the transformation

       @return

        x_Transformed_train: 

        x_Transformed_test: 

    '''
    '''

    3.2 larger transfomred datasets


    '''

    x_Transformed_train = np.array([assign2_utils.random_deform(x,degree,strength) for x in train_dataset])
    x_Transformed_test = np.array([assign2_utils.random_deform(x,degree,strength) for x in test_dataset])
    return x_Transformed_train, x_Transformed_test

    

    '''

#    5. Create the positive and negative pairs

#    (the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 )

    '''



def reshape_input(image_width, image_height, num_channels, pairs):
    #reshape the image as 28*28 
    Twin1_input_pair = np.array([np.reshape(x, (image_width, image_height)) for x in pairs[:, 0]])
    #shape the input for twin1 neural network
    Twin1_input_pair = Twin1_input_pair.reshape(Twin1_input_pair.shape[0],Twin1_input_pair.shape[1],pairs.shape[2], num_channels)
    #reshape the image as 28*28 
    Twin2_input_pair = np.array([np.reshape(x, (image_width, image_height)) for x in pairs[:, 1]])
    #shape the input for twin2 neural network
    Twin2_input_pair = Twin2_input_pair.reshape(Twin2_input_pair.shape[0],Twin2_input_pair.shape[1],pairs.shape[2], num_channels)
   
    return  Twin1_input_pair,  Twin2_input_pair



#image_width 
image_width = 28
#image_height
image_height = 28
# greyscale = 1 channel
num_channels = 1
#-------------------
def preprocess_Data(degree, strength):
    '''
    Define the planned experiments here

    1. load the data

    2. transformed data

    3. Create the positive and negative pairs

    4. run experiment

    5. output report   

    

       @param

      : 

       @return

      :
    '''    
    #   1. load the data
    x_original_train, y_train, x_original_test, y_test = assign2_utils.load_dataset()
    #2. transformed data
    if (degree != 0 and strength !=0 ):
        train_pairs, train_target, val_pairs, val_target  = transform_dataset(x_original_train, x_original_test, degree, strength)
    else:
        train_pairs, train_target, val_pairs, val_target  = x_original_train, y_train, x_original_test, y_test


    # 3. Create the positive and negative pairs
    # the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 

    digit_indicesTrain = [np.where(train_target == i)[0] for i in range(10)]
    train_pairs, train_y = create_pairs(train_pairs, digit_indicesTrain)
    #
    digit_indicesTest = [np.where(val_target == i)[0] for i in range(10)]
    val_pairs, val_y = create_pairs(val_pairs, digit_indicesTest)

    # 4. reshape the pairs for 2D  convolutional layer
    train_twin1_input, train_twin2_input = reshape_input(image_width, image_height, num_channels, train_pairs)
    val_twin1_input, val_twin2_input = reshape_input(image_width, image_height, num_channels, train_pairs)
    return train_twin1_input, train_twin2_input, val_twin1_input, val_twin2_input
#------------------------------------------------------------------------------

def solution(numberOfsolution,degree, strength, percentage):
    
    def first_Architecture(): 
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
    
    def second_Architecture(): 
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
    
    if(numberOfsolution == 1): 
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


if __name__=='__main__':
    solution()


    



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