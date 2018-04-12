from model import Model
import tensorflow as tf
from utils import plot_alignments
from config import LOG_DIR, CHECK_VALS
from tqdm import tqdm

def train():
    g = Model()
    print("Graph for training loaded.")
    
    with tf.Graph().as_default():
        # Note: save_summary_secs - number of seconds between computation of summarys
        # for event log is defaulted to 60 seconds
        # save_model_secs - number of seconds between the creation of model checkpoints
        # is defaulted to 600 seconds

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Run the session
        for i in tqdm(range(g.num_batch)):
            _, g_step = sess.run([g.opt_train, g.global_step])

            # write checkpoint files
            if g_step % CHECK_VALS == 0:
                # save
                #sv.saver.save(sess, LOG_DIR + '/step' + str(g_step))

                # plot the first alignment for logging
                align = sess.run(g.alignments)
                plot_alignments(align[0], g_step)


if __name__ == '__main__':
    train()
    print("Done")