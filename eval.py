#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:02:41 2018

@author: miller
"""

import numpy as np
from dataload import input_load
import tensorflow as tf
from train import Model
from utils import create_spectrograms
from config import LOG_DIR

def eval(): 
    # Load graph
    g = Model(mode="eval"); print("Evaluation Graph loaded")

    # Load data
    fpaths, text_lengths, texts = input_load(mode="eval")

    # Parse
    text = np.fromstring(texts[0], np.int32) # (None,)
    fname, mel, mag = create_spectrograms(fpaths[0])

    text = np.expand_dims(text, 0) # (1, None)
    mels = np.expand_dims(mel, 0) # (1, None, n_mels*r)
    mags = np.expand_dims(mag, 0) # (1, None, n_mfccs)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR)); print("Restored!")

        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # Feed Forward
        ## mel
        mels_hat = np.zeros((1, mels.shape[1], mels.shape[2]), np.float32)  # LOG_DIR.n_mels*LOG_DIR.r
        for i in range(mels.shape[1]):
            _mels_hat = sess.run(g.mel_hat, {g.txt: text, g.mels: mels_hat})
            mels_hat[:, i, :] = _mels_hat[:, i, :]

        ## mag
        merged, gs = sess.run([g.merged, g.global_step], {g.txt:text, g.mels:
            mels, g.mel_hat: mels_hat, g.mags: mags})
        writer.add_summary(merged, global_step=gs)
        writer.close()


if __name__ == '__main__':
    eval()
    print("Done")
    