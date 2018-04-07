"""
networks.py 

Includes the encoder and decoder RNNS
"""
import tensorflow as tf 
import config
from cbhg import cbhg_helper

def encoder(inputs, is_training=True, scope="encoder"):
    # Get encoder/decoder inputs
    self.encoder_inputs = embed(self.x, VOCAB_SIZE, EMBED_SIZE) # (N, T_x, E)
    # Networks
    with tf.variable_scope(scope):
        # Encoder pre-net
        prenet_out = prenet(inputs, is_training=is_training) # (N, T_x, E/2)
        # Run CBHG
        return cbhg_helper(inputs=prenet_out, length=EMBED_SIZE//2, is_running=is_training)

def embed(inputs, vocab_size, num_units, scope="embedding"):
    """ embeds text into a vector """ 
    with tf.variable_scope(scope):
         lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)

def pre-net(inputs, is_training=True, num_hidden_units=None, scope="Pre-Net"): 
    """ 
    apply a set of non-linear transformations, collectively called a "pre-net",
    to each embedding
    """
    if num_hidden_units is None:
        num_hidden_units = [EMBED_SIZE, EMBED_SIZE // 2]
    
    outputs = inputs
    for i in range(len(num_hidden_units):
        outputs = tf.layers.dense(outputs, units=num_units[0], activation=tf.nn.relu, name=("dense" + str(i))
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name=("dropout" + str(i))
    return outputs
