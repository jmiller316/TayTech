"""
cbhg.py

Contains functions for the CBHG module used in the encoder and after
the decoder. 
"""
import tensorflow as tf
from config import *

def cbhg(inputs, lengths, is_training, scope="cbhb", K=16, projections=[EMBED_SIZE//2, EMBED_SIZE//2]):
    """
    The CBGB Module used to process the inputs
    Parameters:
        inputs - the inputs to process
        length - the length of the inputs
        is_training - whether or not the graph is training
        scope - used for tf.variable_scope
        K - a parameter used in generating the convolutional banks
        projections - size of projections
    Returns:
        rnn - a rnn with GRU cells for the graph
    """
    with tf.variable_scope(scope):
        # Convolutional bank and max pool
        # The input sequence is first convolved with K sets of 1-D convolutional filters.
        banks = conv1d_banks(inputs, K=K, is_training=is_training) 
        banks = tf.layers.max_pooling1d(banks, pool_size=2, strides=1, padding="same")
        # Conv1D Layers
        banks = conv1d(banks, kernel_size=3, scope="conv1d_1", activation_fn=tf.nn.relu, is_training=is_training) 
        banks = conv1d(banks, kernel_size=3, scope="conv1d_2", activation_fn=None, is_training=is_training)

        # Multi-layer highway network
        highway_in = inputs + banks
        
        for i in range(0,NUM_HIGHWAY_LAYERS):
            highway_in = highwaynet(highway_in, num_units=EMBED_SIZE//2, scope=("highway_net" + str(i)))

        # bidirectional GRU RNN to extract sequential features fromboth
        # forward and backward context
        rnn = cbhg_rnn(inputs,lengths, scope=("cbgh_gru_rnn_" + scope))
    return rnn

def cbhg_helper(inputs, lengths, is_training, post=False):
    """
    Helper function for the CBHG module. Specifies different parameters based
    off of the value of post.

    Parameters:
        input - input to process
        lengths - length of inputs
        is_training - whether or not the graph is training
        post - whether to run the post-network cbhg or the encoder cbhg
    """
    if post:
        return cbhg(inputs, None, is_training, scope='post_cbhg', K=8, 
                                  projections=[EMBED_SIZE,lengths])
    return cbhg(inputs, lengths, is_training, scope='pre_cbhg', K=16, 
                                 projections=[EMBED_SIZE//2, EMBED_SIZE//2])
    
def cbhg_rnn(inputs, input_length, scope="cbgh_rnn"):
    """
    Create the RNN with GRUCells for the CBHG module
    Returns the bidirectional rnn
    """
    with tf.variable_scope(scope):
        # The bidirectional GRU RNN
        gru_rnn = tf.nn.bidirectional_dynamic_rnn(
          tf.contrib.rnn.GRUCell(EMBED_SIZE//2),
          tf.contrib.rnn.GRUCell(EMBED_SIZE//2),
          inputs,
          sequence_length=input_length,
          dtype=tf.float32)
    return gru_rnn

def conv1d(inputs, kernel_size, activation_fn=None, is_training=True, scope="conv1d", filters=None):
    """
    Create the 1-D convolutional layers and normalize.
    Parameters:
        inputs - the input features
        kernel_size - the size of hte kernel in the conv1d
        activation_fn - the activation function for the conv1d
        is_training - whether or not the graph is training
        scope - the scope for variable_scope
        filters - filter sizes
    Returns:
        A normalized conv1d layer
    Notes:
        Each convolutional layer is batch normalized
    """
    # If no input filters
    if filters is None:
        filters = inputs.get_shape().as_list[-1]
    with tf.variable_scope(scope):
        # Create the conv1d
        conv1d_output = tf.layers.conv1d(inputs, 
                filters=filters,kernel_size=kernel_size,
                activation=activation_fn, padding='same')
    # Batch normalization is used for all convolutional layers
    return tf.layers.batch_normalization(conv1d_output, training=is_training)

def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks"):
    """
    This function convolves the input sequence with K sets of 1-D convolutional filters,
    where the k-th set contains C_k filters of width k. 
    These filters explicitly model local and contextual information.
    Parameters:
        inputs - the input sequence
        K - the number of sets of convoutional filters
        is_training - whether or not the model is training
        scope - the scope for tf.variable_scope
    Returns:
        The outputs after the convolutional bank
    """
    with tf.variable_scope(scope):
        # The first convolutional filter
        outputs = conv1d(inputs,  filters=EMBED_SIZE//2, kernel_size=1, 
                                    is_training=is_training, scope="conv1d_convbanks1") 
        
        # The next filters until there are k filters
        for k in range(2, K+1):
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, filters=EMBED_SIZE // 2, kernel_size=k, 
                                    is_training=is_training, scope="conv1d_convbanks2")
                outputs = tf.concat((outputs, output), -1)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    return outputs 

def highwaynet(inputs, num_units=None, scope="highwaynet"):
    """
    One layer of the highway network for the CBHG module.
    Returns:
        The outputs after one layer of the highway network.
    Notes:
        outputs = H ∗ T + inputs ∗ (1−T)
        Where H is a highway network consisting of multiple blocks 
        and T is a transform gate output.
    """
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope):
        # Values used to construct highway network layer
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="R2D2")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="D2R2")
        
        # Highway network layer
        outputs = H*T + inputs*(1.-T)
    return outputs
