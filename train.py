from model import Model
import tensorflow as tf
from utils import plot_alignments, spectrogram2wav
from config import CHECK_VALS, LOG_DIR, SR
import numpy as np
import librosa
import os
from tqdm import tqdm
import time

def train():
    # Stats
    time_count = 0
    time_sum = 0
    loss_count = 0
    loss_sum = 0

    check_path = os.path.join(LOG_DIR, 'model')
    check_path2 = os.path.join(LOG_DIR, 'model.meta-320.meta')

    g = Model()
    print("Graph for training loaded.")

    # Session
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.isfile(check_path):
            saver = tf.train.import_meta_graph(check_path)
            saver.restore(sess, tf.train.latest_checkpoint('./'))

        # run and initialize
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        # Run the session
        for i in tqdm(range(g.num_batch)):
            start_time = time.time()
            g_step, g_loss, g_opt = sess.run([g.global_step, g.loss, g.opt_train])

            # Generate stats
            time_count += 1
            loss_count += 1
            time_sum += time.time() - start_time
            loss_sum += g_loss

            message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % \
                      ( g_step, time_sum/time_count, g_loss, loss_sum/loss_count)
            print(message)

            if g_step % CHECK_VALS == 0:
                print("Saving checkpoint to %s at step %d" %(check_path, g_step))
                saver.save(sess, check_path, global_step=g_step)

                # Saving the audio and alignment
                print('Saving audio and alignment...')
                audio_out, alignments = sess.run([g.audio_out, g.alignments[0]])

                # The wav file
                librosa.output.write_wav(os.path.join(LOG_DIR, 'step-%d-audio.wav' % g_step),
                                        audio_out, SR)

                # plot alignments
                plot_alignments(alignments, global_step=g_step)

    sess.close()


if __name__ == '__main__':
    train()
    print("Done")
