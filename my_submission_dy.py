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
#K.set_image_dim_ordering('tf')

import assign2_utils

sess = tf.Session()

#------------------------------------------------------------------------------
# Define parameters:
#------------------------------------------------------------------------------
# image dimensions    
image_width   = 28
image_height  = 28
num_channels  = 1  # greyscale = 1 channel
input_shape = (image_width, image_height, num_channels) #input of the architecture

NumberOfClass = 10
batch_size=128
epoch = 1

positive_pair = 1
negative_pair = 0
# Experiment Architecture
architecture_flag= 1# which network architecture to use
architecture_flag= 2
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

    y_positive_true = 1
    y_negative_true = 0 
    return K.mean(y_true * K.square(y_pred) +
                  (y_positive_true - y_true) * K.square(K.maximum(margin - y_pred, y_negative_true)))

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
    threshold = 0.5
    n = labels.shape[0]

    acc = (labels[predictions.ravel() < threshold].sum() + # count True Positive
                   (positive_pair-labels[predictions.ravel() >= threshold]).sum())/ n # True Negative
    return acc


#------------------------------------------------------------------------------

def create_pairs(x, digit_indices, percentage = 1):
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
    FirstClass = 1
    totalNumberOfClass = 10
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(totalNumberOfClass)]) - 1
    #to create the 100,000 training dataset control the number of n
    MaxNumberOfPair=5000
    if(percentage !=1):
        n = int(MaxNumberOfPair*percentage)
    if(n>MaxNumberOfPair):
        n = MaxNumberOfPair
    for d in range(totalNumberOfClass):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # z1 and z2 form a positive pair
            inc = random.randrange(FirstClass, totalNumberOfClass)
            dn = (d + inc) % totalNumberOfClass
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # z1 and z2 form a negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [positive_pair, negative_pair]
    return np.array(pairs), np.array(labels)   


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
    x_Transformed_train = np.array([assign2_utils.random_deform(x,degree,strength) for x in train_dataset])
    x_Transformed_test = np.array([assign2_utils.random_deform(x,degree,strength) for x in test_dataset])
    
    return x_Transformed_train, x_Transformed_test
#------------------------------------------------------------------------------
def reshape_input(image_width, image_height, num_channels, train_pairs):
    ''' 
    This function will reshape input for 2D Convolution:

       @param
        image_width, image_height, num_channels : defined at top of code
        pairs : 

       @return
        Twin1_input_pair: 
        Twin2_input_pair: 
    '''
    Twin1_input_pair = train_pairs[:,0].reshape(train_pairs[:,0].shape[0],train_pairs[:,0].shape[1], train_pairs[:,0].shape[2], num_channels)
    Twin2_input_pair = train_pairs[:,1].reshape(train_pairs[:,1].shape[0],train_pairs[:,1].shape[1], train_pairs[:,1].shape[2], num_channels)
    
    return Twin1_input_pair, Twin2_input_pair


def split_Data(dataset, target, percentage):
    theNumberOfSampleSize = 5000
    n= int(theNumberOfSampleSize*percentage) + 1
    digit_indices = [np.where(target == i)[0] for i in range(NumberOfClass)]
    split1 =np.reshape([digit_indices[d][:n] for d in range(NumberOfClass)], NumberOfClass*n)
    twin1_input1 = dataset[split1]
    split2 =np.reshape([digit_indices[d][n:theNumberOfSampleSize+2] for d in range(NumberOfClass)], NumberOfClass*(theNumberOfSampleSize-n+2))
    twin1_input2  = dataset[split2]
    return twin1_input1, twin1_input2, split1, split2


#------------------------------------------------------------------------------
def noSplit_preprocess_Data(degree, strength):
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
    #   1. load the data
    x_original_train, train_target, x_original_test, val_target = assign2_utils.load_dataset()
    
    x_original_train = x_original_train.astype('float32')
    x_original_test = x_original_test.astype('float32')
    
    x_original_train /= 255 # normalized the entries between 0 and 1
    x_original_test /= 255
    
    
    #   2. transformed data
    if (degree != 0 or strength !=0 ):
        train_pairs, val_pairs = transform_dataset(x_original_train, x_original_test, degree, strength)
    else:
        train_pairs, val_pairs = x_original_train, x_original_test

    #   3. Create the positive and negative pairs
    #   the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 

    digit_indicesTrain = [np.where(train_target == i)[0] for i in range(10)]
    digit_indicesTest = [np.where(val_target == i)[0] for i in range(10)]

    train_pairs, train_y = create_pairs(train_pairs, digit_indicesTrain)
    val_pairs, val_y = create_pairs(val_pairs, digit_indicesTest)
        
        #   4. reshape the pairs for 2D  convolutional layer
    train_twin1_input, train_twin2_input = reshape_input(image_width, image_height, num_channels, train_pairs)
    val_twin1_input, val_twin2_input = reshape_input(image_width, image_height, num_channels, val_pairs)
    
    return train_twin1_input, train_twin2_input, val_twin1_input, val_twin2_input, train_y, val_y
    
    
def split_preprocess_Data(degree1, degree2, strength1, strength2, percentage =1):     
    #   1. load the data
    
    x_original_train, train_target, x_original_test, val_target = assign2_utils.load_dataset()
    #train_pairs, val_pairs = transform_dataset(x_original_train, x_original_test, 30, 0.15)
    # 2. split the datset.
    x_original_train = x_original_train.astype('float32')
    x_original_test = x_original_test.astype('float32')
    
    normaliseValue = 255
    x_original_train /= normaliseValue # normalized the entries between 0 and 1
    x_original_test /= normaliseValue
    
    x_original_train1, x_original_train2, train_target1, train_target2  = split_Data(x_original_train,train_target, percentage)
    #x_original_test1, x_original_test2 , val_target1, val_target2  = split_Data(x_original_test,val_target, percentage)

    # 3. transform the dataset.
    warped_pairs1 = np.array([assign2_utils.random_deform(x, degree1, strength1) for x in x_original_train1])
    warped_pairs2 = np.array([assign2_utils.random_deform(x, degree2, strength2) for x in x_original_train2])
    # train_pairs2 = np.array([assign2_utils.random_deform(x, degree2, strength2) for x in x_original_train2]
    

    digit_indicesTrain1 = [np.where(train_target[train_target1] == i)[0] for i in range(NumberOfClass)]
    train_pairs1, train_y1 = create_pairs(warped_pairs1, digit_indicesTrain1, percentage)
    digit_indicesTrain2 = [np.where(train_target[train_target2] == i)[0] for i in range(NumberOfClass)]
    train_pairs2, train_y2 = create_pairs(warped_pairs2, digit_indicesTrain2,  percentage)
    

    #   4. reshape the pairs for 2D  convolutional layer
    
    train_twin1_input1, train_twin2_input1 = reshape_input(image_width, image_height, num_channels, train_pairs1)
    train_twin1_input2, train_twin2_input2 = reshape_input(image_width, image_height, num_channels, train_pairs2)

    return train_twin1_input1, train_twin2_input1, train_twin1_input2, train_twin2_input2, train_y1, train_y2
              # val_twin1_input1, val_twin2_input1, val_twin1_input2, val_twin2_input2, val_y1, val_y2

#------------------------------------------------------------------------------

def solution(architecture_flag = 1, degree1 = 0, degree2 = 0, strength1 = 0, strength2 =0, percentage = 1):
    def Kal_initial_Architecture(input_shape):
        seq = keras.models.Sequential()
        #first convolutional layer
        conv1_filter = 96
        conv1_kernel_size =(11, 11)
        conv1_stride = (1, 1)
        #first pooling layer
        pool1_size = (3,3)
        pool1_stride = (1, 1)
        #second convolutional layer
        conv2_filter = 256
        conv2_kernel_size =(5, 5)
        conv2_stride = (1, 1)
        #second pooling layer
        pool2_size = (3,3)
        pool2_stride = (2, 2)
        #third convolutional layer
        conv3_filter = 384
        conv3_kernel_size =(2, 2)
        conv3_stride = (1, 1)
        #forth convolutional layer
        conv4_filter = 384
        conv4_kernel_size =(3, 3)
        conv4_stride = (1, 1)
        #fitth convolutional layer
        conv5_filter = 256
        conv5_kernel_size =(3, 3)
        conv5_stride = (1, 1)
        #fully connected layer output1,2
        fully_connected_layer_output =4096
        dropout1_probability = 0.5
        #fully connected layer output
        #dropout2
        dropout2_probability = 0.5

        #'tf' mode is it at index 3 (e.g. 256, 256, 3)
        #first convolutional layer
        seq.add(keras.layers.Conv2D(conv1_filter, kernel_size=conv1_kernel_size,strides= conv1_stride , activation='relu', input_shape=input_shape))
        seq.add(keras.layers.BatchNormalization())
        #first pooling layer
        seq.add(keras.layers.MaxPooling2D(pool_size=pool1_size,strides=pool1_stride))
        seq.add(keras.layers.Activation('relu'))
        #second convolutional layer
        seq.add(keras.layers.Conv2D(conv2_filter, kernel_size=conv2_kernel_size,strides= conv2_stride, activation='relu'))
        seq.add(keras.layers.BatchNormalization())
        #second pooling layer
        seq.add(keras.layers.MaxPooling2D(pool_size=pool2_size,strides=pool2_stride, data_format="channels_last", padding="same"))
        seq.add(keras.layers.Activation('relu'))
        #third convolutional layer
        seq.add(keras.layers.Conv2D(conv3_filter, kernel_size=conv3_kernel_size,strides= conv3_stride, activation='relu'))
        #forth convolutional layer
        seq.add(keras.layers.Conv2D(conv4_filter, kernel_size=conv4_kernel_size,strides= conv4_stride, activation='relu'))
        #fifth convolutional layer
        seq.add(keras.layers.Conv2D(conv5_filter, kernel_size=conv5_kernel_size,strides= conv5_stride, activation='relu'))
        seq.add(keras.layers.Flatten())
        #fully_connect layer1
        seq.add(keras.layers.Dense(fully_connected_layer_output , activation='relu'))
        #dropout1
        seq.add(keras.layers.Dropout(dropout1_probability))
        #fully_connect layer2
        seq.add(keras.layers.Dense(fully_connected_layer_output , activation='relu'))
        #dropout2
        seq.add(keras.layers.Dropout(dropout2_probability))
        return seq
    
    def second_Architecture(input_shape):
        seq = keras.models.Sequential()
        conv1_filter = 256
        conv1_kernel_size =(5, 5)
        conv1_stride = (1, 1)
        #second pooling layer
        pool1_size = (3,3)
        pool1_stride = (2, 2)
        #third convolutional layer
        conv2_filter = 384
        conv2_kernel_size =(2, 2)
        conv2_stride = (1, 1)
        #forth convolutional layer
        conv3_filter = 384
        conv3_kernel_size =(3, 3)
        conv3_stride = (1, 1)
        #fitth convolutional layer
        #fully connected layer output1,2
        fully_connected_layer_output =4096
        dropout1_probability = 0.5
        #fully connected layer output
        #dropout2
        dropout2_probability = 0.5


        #first convolutional layer
        seq.add(keras.layers.Conv2D(conv1_filter, kernel_size=conv1_kernel_size,strides= conv1_stride, activation='relu', input_shape=input_shape))
        seq.add(keras.layers.BatchNormalization())
        #second pooling layer
        seq.add(keras.layers.MaxPooling2D(pool_size=pool1_size,strides=pool1_stride, data_format="channels_last", padding="same"))
        seq.add(keras.layers.Activation('relu'))
        #third convolutional layer
        seq.add(keras.layers.Conv2D(conv2_filter, kernel_size=conv2_kernel_size,strides= conv2_stride, activation='relu'))
        #forth convolutional layer
        seq.add(keras.layers.Conv2D(conv3_filter, kernel_size=conv3_kernel_size,strides= conv3_stride, activation='relu'))
        #fifth convolutional layer
        seq.add(keras.layers.Flatten())
        #fully_connect layer1
        seq.add(keras.layers.Dense(fully_connected_layer_output , activation='relu'))
        #dropout1
        seq.add(keras.layers.Dropout(dropout1_probability))
        #fully_connect layer2
        seq.add(keras.layers.Dense(fully_connected_layer_output , activation='relu'))
        #dropout2
        seq.add(keras.layers.Dropout(dropout2_probability))
        return seq
    
    if(percentage !=0):
        train_twin1_input1, train_twin2_input1, train_twin1_input2, train_twin2_input2, train_y1, train_y2  = split_preprocess_Data(degree1, degree2, strength1, strength2, percentage =1)
    else:
        train_twin1_input, train_twin2_input, val_twin1_input, val_twin2_input, train_y, val_y = noSplit_preprocess_Data(degree1, strength1)
    
    
    if(architecture_flag == 1):
        base_network = Kal_initial_Architecture(input_shape)
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
#    train_siamese(model, tr_pairs, tr_y, te_pairs, te_y)

    
    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    
    if(percentage !=0):
        model.fit([train_twin1_input1, train_twin2_input1], train_y1,
        batch_size=batch_size,
        epochs=epoch)
        model.fit([train_twin1_input2, train_twin2_input2], train_y2,
        batch_size=batch_size,
        epochs=epoch)
    
        train_twin1_input = np.concatenate((train_twin1_input1, train_twin1_input2),axis = 0)
        train_twin2_input = np.concatenate((train_twin2_input1, train_twin2_input2),axis = 0)
        
        train_y = np.concatenate((train_y1, train_y2),axis = 0)
    else:
        model.fit([train_twin1_input, train_twin2_input], train_y,batch_size=batch_size,
        epochs=epoch,
        validation_data=([val_twin1_input, val_twin2_input], val_y))
    
    
# compute final accuracy on training and test sets
    pred = model.predict([train_twin1_input, train_twin2_input])
    tr_acc = compute_accuracy(pred, train_y)
    pred = model.predict([val_twin1_input, val_twin2_input])
    te_acc = compute_accuracy(pred, val_y)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

def Kal_initial_Architecture(input_shape):
    seq = keras.models.Sequential()
    #first convolutional layer
    conv1_filter = 96
    conv1_kernel_size =(11, 11)
    conv1_stride = (1, 1)
    #first pooling layer
    pool1_size = (3,3)
    pool1_stride = (1, 1)
    #second convolutional layer
    conv2_filter = 256
    conv2_kernel_size =(5, 5)
    conv2_stride = (1, 1)
    #second pooling layer
    pool2_size = (3,3)
    pool2_stride = (2, 2)
    #third convolutional layer
    conv3_filter = 384
    conv3_kernel_size =(2, 2)
    conv3_stride = (1, 1)
    #forth convolutional layer
    conv4_filter = 384
    conv4_kernel_size =(3, 3)
    conv4_stride = (1, 1)
    #fitth convolutional layer
    conv5_filter = 256
    conv5_kernel_size =(3, 3)
    conv5_stride = (1, 1)
    #fully connected layer output1,2
    fully_connected_layer_output =4096
    dropout1_probability = 0.5
    #fully connected layer output
    #dropout2
    dropout2_probability = 0.5

    #'tf' mode is it at index 3 (e.g. 256, 256, 3)
    #first convolutional layer
    seq.add(keras.layers.Conv2D(conv1_filter, kernel_size=conv1_kernel_size,strides= conv1_stride , activation='relu', input_shape=input_shape))
    seq.add(keras.layers.BatchNormalization())
    #first pooling layer
    seq.add(keras.layers.MaxPooling2D(pool_size=pool1_size,strides=pool1_stride))
    seq.add(keras.layers.Activation('relu'))
    #second convolutional layer
    seq.add(keras.layers.Conv2D(conv2_filter, kernel_size=conv2_kernel_size,strides= conv2_stride, activation='relu'))
    seq.add(keras.layers.BatchNormalization())
    #second pooling layer
    seq.add(keras.layers.MaxPooling2D(pool_size=pool2_size,strides=pool2_stride, data_format="channels_last", padding="same"))
    seq.add(keras.layers.Activation('relu'))
    #third convolutional layer
    seq.add(keras.layers.Conv2D(conv3_filter, kernel_size=conv3_kernel_size,strides= conv3_stride, activation='relu'))
    #forth convolutional layer
    seq.add(keras.layers.Conv2D(conv4_filter, kernel_size=conv4_kernel_size,strides= conv4_stride, activation='relu'))
    #fifth convolutional layer
    seq.add(keras.layers.Conv2D(conv5_filter, kernel_size=conv5_kernel_size,strides= conv5_stride, activation='relu'))
    seq.add(keras.layers.Flatten())
    #fully_connect layer1
    seq.add(keras.layers.Dense(fully_connected_layer_output , activation='relu'))
    #dropout1
    seq.add(keras.layers.Dropout(dropout1_probability))
    #fully_connect layer2
    seq.add(keras.layers.Dense(fully_connected_layer_output , activation='relu'))
    #dropout2
    seq.add(keras.layers.Dropout(dropout2_probability))
    return seq
#------------------------------------------------------------------------------
if __name__=='__main__':

    
    '''
    architecture_flag:1 for the initial architecture. 2 is for the second architecutre
    degree1:
    degree2: 0 
    strength1: 
    strength2: 
    percentage:
    '''
    x_original_train, train_target, x_original_test, val_target = assign2_utils.load_dataset()
    #train_pairs, val_pairs = transform_dataset(x_original_train, x_original_test, 30, 0.15)
    # 2. split the datset.
    x_original_train = x_original_train.astype('float32')
    x_original_test = x_original_test.astype('float32')
    
    normaliseValue = 255
    x_original_train /= normaliseValue # normalized the entries between 0 and 1
    x_original_test /= normaliseValue
    
    x_original_train1, x_original_train2, train_target1, train_target2  = split_Data(x_original_train,train_target, 0.4)
    #x_original_test1, x_original_test2 , val_target1, val_target2  = split_Data(x_original_test,val_target, percentage)

    # 3. transform the dataset.
    warped_pairs1 = np.array([assign2_utils.random_deform(x, 15, 0.1) for x in x_original_train1])
    warped_pairs2 = np.array([assign2_utils.random_deform(x, 45, 0.3) for x in x_original_train2])
    # train_pairs2 = np.array([assign2_utils.random_deform(x, degree2, strength2) for x in x_original_train2]
    

    digit_indicesTrain1 = [np.where(train_target[train_target1] == i)[0] for i in range(NumberOfClass)]
    train_pairs1, train_y1 = create_pairs(warped_pairs1, digit_indicesTrain1, 0.4)
    digit_indicesTrain2 = [np.where(train_target[train_target2] == i)[0] for i in range(NumberOfClass)]
    train_pairs2, train_y2 = create_pairs( warped_pairs2, digit_indicesTrain2,  0.6)
    
    
    #   4. reshape the pairs for 2D  convolutional layer
    
    train_twin1_input1, train_twin2_input1 = reshape_input(image_width, image_height, num_channels, train_pairs1)
    train_twin1_input2, train_twin2_input2 = reshape_input(image_width, image_height, num_channels, train_pairs2)

    base_network = Kal_initial_Architecture(input_shape)
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
    # train
    model = keras.models.Model([input_a, input_b], distance)
    rms = keras.optimizers.RMSprop()

    model.compile(loss=contrastive_loss, optimizer=rms)
     
    # Our model take as input a pair of images input_a and input_b
    # and output the Euclidian distance of the mapped inputs

    
    model.fit([train_twin1_input1, train_twin2_input1], train_y1,
    batch_size=batch_size,
    epochs=epoch)
    model.fit([train_twin1_input2, train_twin2_input2], train_y2,
    batch_size=batch_size,
    epochs=epoch)

    train_twin1_input = np.concatenate((train_twin1_input1, train_twin1_input2),axis = 0)
    train_twin2_input = np.concatenate((train_twin2_input1, train_twin2_input2),axis = 0)
    
    train_y = np.concatenate((train_y1, train_y2),axis = 0)
    
        
    # compute final accuracy on training and test sets
    pred = model.predict([train_twin1_input, train_twin2_input])
    tr_acc = compute_accuracy(pred, train_y)

    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    