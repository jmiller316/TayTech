"""
utils.py

Miscellaneous functions that we will need.
"""

"""
attention
"""
def attention(a, b, num_units=0):
    return "a", "b"

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
    if global_step < 500000:
        return 0.001
    if global_step >= 500000 and global_step < 1000000:
        return 0.005
    if global_step >= 1000000 and global_step < 2000000:
        return 0.0003
    return 0.0001
