"""
networks.py 

Includes the encoder and decoder RNNS
"""
import tensorflow as tf 
from config import *
from cbhg import cbhg_helper

def encoder(inputs, is_training=True, scope="encoder"):
    """
    Encoder for the input sequence.
    Embeds the character sequence -> Runs through the pre-net -> CBHG Module
        
    """
    # Get encoder/decoder inputs
    encoder_inputs = embed(inputs, VOCAB_SIZE, EMBED_SIZE) # (N, T_x, E)
    # Networks
    with tf.variable_scope(scope):
        # Encoder pre-net
        pre_out = pre_net(encoder_inputs, is_training=is_training) # (N, T_x, E/2)
        # Run CBHG
        cbhg_net = cbhg_helper(inputs=pre_out, lengths=EMBED_SIZE//2, is_training=is_training)
    return cbhg_net

def embed(inputs, vocab_size, num_units, scope="embedding"):
    """
    Embeds character sequence into a continuous vector
    """ 
    with tf.variable_scope(scope):
        # Create a look-up table
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        
        # Concatenate the tensors along one-dimension
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)

def pre_net(inputs, is_training=True, num_hidden_units=None, scope="Pre-Net"): 
    """ 
    Apply a set of non-linear transformations, collectively called a "pre-net",
    to each embedding.
    """
    if num_hidden_units is None:
        num_hidden_units = [EMBED_SIZE, EMBED_SIZE // 2]
    
    # Apply the series of transformations
    outputs = inputs
    for i in range(len(num_hidden_units)):
        outputs = tf.layers.dense(outputs, units=num_hidden_units[0], activation=tf.nn.relu, name=("dense" + str(i)))
        outputs = tf.layers.dropout(outputs, rate=DROPOUT_RATE, training=is_training, name=("dropout" + str(i)))
    return outputs
