"""
utils.py

Miscellaneous functions that we will need.
"""
import tensorflow as tf


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.0):
    """
    Learning_rate_decay.

    The learning rate decay starts from 0.001 and is reduced to 0.0005, 0.0003, 
    and 0.0001 after 500K, 1M, and 2M global steps respectively.

    While the rate is hardcoded in here, it can be changed to the Naome from 
    tensor2tensor if it does not work well. 
    """
    """# Naomi from tensor2tensor
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**(-1.5), step**(-0.5))"""
    # Constants for the learning rate
    r1 = tf.constant(0.001)
    r2 = tf.constant(0.0005)
    r3 = tf.constant(0.0003)
    r4 = tf.constant(0.0001)
    l1 = tf.constant(500000)
    l2 = tf.constant(1000000)
    l3 = tf.constant(2000000)
    
    # Compare the values to determine the learning rate
    if tf.less(global_step, l1):
        return r1
    if tf.greater_equal(global_step, l1) and tf.less(global_step, l2):
        return r2
    if tf.greater_equal(global_step, l2) and tf.less(global_step, l3):
        return r3
    return r4
