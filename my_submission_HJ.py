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


image_width = 28
image_height = 28
num_channels = 1 # greyscale = 1 channel


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
    
def simplistic_solution():
    '''
    
    Train a Siamese network to predict whether two input images correspond to the 
    same digit.
    
    WARNING: 
        in your submission, you should use auxiliary functions to create the 
        Siamese network, to train it, and to compute its performance.
    
    
    '''
    def create_simplistic_base_network(input_dim):
        '''
        Base network to be shared (eq. to feature extraction).
        '''
        seq = keras.models.Sequential()
        seq.add(keras.layers.Dense(128, input_shape=(input_dim,), activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
        return seq
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
    # load the dataset
    x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()

    # Example of magic numbers (6000, 784)
    # This should be avoided. Here we could/should have retrieve the
    # dimensions of the arrays using the numpy ndarray method shape 
    x_train = x_train.reshape(60000, 784) 
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 # normalized the entries between 0 and 1
    x_test /= 255
    input_dim = 784 # 28x28

    #

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    train_pairs, train_y = create_pairs(x_train, digit_indices)
    
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    val_pairs, val_y = create_pairs(x_test, digit_indices)
    
    # network definition
    base_network = create_simplistic_base_network(input_dim)
    
    input_a = keras.layers.Input(shape=(input_dim,))
    input_b = keras.layers.Input(shape=(input_dim,))
    
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

    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
              validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y))

    # compute final accuracy on training and test sets
    pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, train_y)
    pred = model.predict([val_pairs[:, 0], val_pairs[:, 1]])
    te_acc = compute_accuracy(pred, val_y)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


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
    Twin1_input_pair = Twin1_input_pair.reshape(Twin1_input_pair.shape[0],Twin1_input_pair2.shape[1],pairs.shape[2], num_channels)
    
    #reshape the image as 28*28 
    Twin2_input_pair1 = np.array([np.reshape(x, (image_width, image_height)) for x in pairs[:, 0]])
    #shape the input for twin2 neural network
    Twin2_input_pair1 = Twin1_input_pair2.reshape(Twin2_input_pair.shape[0],Twin2_input_pair.shape[1],pairs.shape[2], num_channels)
    
    
    tr_pair2  = np.array([np.reshape(x, (image_width, image_height), num_channels) for x in tr_pairs[:, 1]])
    tr_pair2 = tr_pair2.reshape(tr_pair2.shape[0],tr_pair2.shape[1],tr_pair2.shape[2], num_channels)
    
    
    te_pair1 = np.array([np.reshape(x, (image_width,image_height), num_channels) for x in te_pairs[:, 0]])
    te_pair1 = te_pair1.reshape(te_pair1.shape[0],te_pair1.shape[1],te_pair1.shape[2], num_channels)
    
    te_pair2= np.array([np.reshape(x, (image_width,image_height), num_channels) for x in te_pairs[:, 1]])
    te_pair2 = te_pair2.reshape(te_pair2.shape[0],te_pair2.shape[1],te_pair2.shape[2], num_channels)



def solution1():
    configure_experiments(0, 0)
#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------        

def configure_experiments(degree, strength):
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

#    1. load the data
    x_original_train, y_train, x_original_test, y_test = assign2_utils.load_dataset()
    
#2. transformed data
    if (degree != 0):
        train_pairs, train_y, val_pairs, val_y  = transform_dataset(x_original_train, x_original_test, degree, strength)
    
    else:
        train_pairs, train_y, val_pairs, val_y  = x_original_train, y_train, x_original_test, y_test
        
        
#    3. Create the positive and negative pairs
#    (the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 )
    digit_indicesTrain = [np.where(train_y == i)[0] for i in range(10)]
    train_pairs, train_y = create_pairs(train_pairs, digit_indicesTrain)
    #
    digit_indicesTest = [np.where(val_y == i)[0] for i in range(10)]
    val_pairs, val_y = create_pairs(val_pairs, digit_indicesTest)
    
# 4. reshape the pairs for 2D convolutional layer


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__=='__main__':
    solution1()
    #    1. load the data
    x_original_train, y_train, x_original_test, y_test = assign2_utils.load_dataset()
    
#2. transformed data
    train_pairs, train_y, val_pairs, val_y  = x_original_train, y_train, x_original_test, y_test 
        
#    3. Create the positive and negative pairs
#    (the total number of pairs, one pair(2 numbers) in the same or not same classification, 28*28 )
    digit_indicesTrain = [np.where(train_y == i)[0] for i in range(10)]
    train_pairs, train_y = create_pairs(train_pairs, digit_indicesTrain)
    #
    digit_indicesTest = [np.where(val_y == i)[0] for i in range(10)]
    val_pairs, val_y = create_pairs(val_pairs, digit_indicesTest)
    

    

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
