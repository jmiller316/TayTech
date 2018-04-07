"""

"""
import tensorflow as tf

def cbhg(inputs, lengths, is_running, scope, K, projections):
    banks = conv1d_banks(prenet_out, K=K, is_training=is_training) 
    banks = tf.layers.max_pooling1d(banks, pool_size=2, strides=1, padding="same")
    banks = conv1d(banks, kernel_size=3, scope="conv1d_1", activation=tf.nn.relu, is_training=is_running) 
    banks = conv1d(banks, kernel_size=3, scope="conv1d_2", activation=none, is_training=is_running)
    highway_in = inputs + banks
    for i in range(0,4):
        highway_in = highwaynet(highway_in,scope=("highway_net" + str(i)))
    return cbgh_rnn(inputs,length, scope=("cbgh_gru_rnn_" + scope))
def cbhg_helper(inputs, length, is_running, post=False):
    if post:
        return cbhg(inputs, None, is_training, scope='post_cbhg', K=8, projections=[256,length])
    return cbhg(inputs, lengths, is_training, scope='pre_cbhg', K=16, projections=[128,128])
    
def cbhg_rnn(inputs, input_length, scope="cbgh_rnn"):
    with tf.variable_scope(scope):
        gru_rnn = tf.nn.bidirectional_dynamic_rnn(
          tf.countrib.rnn.GRUCell(128),
          tf.countrib.rnn.GRUCell(128),
          rnn_input,
          sequence_length=input_lengths,
          dtype=tf.float32)
    return gru_rnn

def conv1d(inputs, filters=None, kernel_size, channels, activation, is_training, scope):
    if filters is None:
        filters = inputs.get_shape().as_list[-1]
    with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=filters,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return ctf.layers.batch_normalization(conv1d_output, training=is_training)

def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, hp.embed_size//2, 1) 
        for k in range(2, K+1):
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, hp.embed_size // 2, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)
    return outputs 

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="R2D2")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="D2R2")
        outputs = H*T + inputs*(1.-T)
    return outputs