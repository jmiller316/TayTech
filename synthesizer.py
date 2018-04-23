#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:30:48 2018

@author: miller
"""

import os
import numpy as np
import tensorflow as tf
import tqdm
from model import Model
import codecs
import librosa
from config import LOG_DIR, N_MELS, REDUCTION_FACTOR, SAVE_DIR, SR, TEST_DATA, ENCODING
from utils import normalize_text, create_vocab, spectrogram2wav

class Synthesizer:
    def __init__(self):
        self.text = None
        self.model = None
        self.mels_hat = None
        self.mags = None

    def synthesize(self, checkpoint_path, text=None):
        print('Constructing model...')
        self.model = Model(mode="synthesize")

        self.load_text(text)
        # Session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Loading checkpoint: %s' % checkpoint_path)
            saver = tf.train.import_meta_graph(checkpoint_path)
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

            # Feed Forward
            ## mel
            self.mels_hat = np.zeros((self.text.shape[0], 200, N_MELS * REDUCTION_FACTOR), np.float32)
            for j in tqdm.tqdm(range(200)):
                feed_dict = {
                    self.model.txt: self.text,
                    self.model.mels: self.mels_hat
                }
                mel_hat2 = sess.run(self.model.mel_hat, feed_dict)
                self.mels_hat[:, j, :] = mel_hat2[:, j, :]

            ## mag
            feed_dict2 = {self.model.mel_hat: self.mels_hat}
            self.mags = sess.run(self.model.mags_hat, feed_dict2)
            for i, mag in enumerate(self.mags):
                print("File {}.wav is being generated ...".format(i + 1))
                audio = spectrogram2wav(mag)
                librosa.output.write_wav(os.path.join(SAVE_DIR, '{}.wav'.format(i + 1)), audio, SR)

    #self.session = tf.Session()
    #self.session.run(tf.global_variables_initializer())
    #saver = tf.train.Saver()
    #saver.restore(self.session, checkpoint_path)

    def load_text(self, text=None):
        # Clean up the text
        if text is None:
            lines = codecs.open(TEST_DATA, 'r', ENCODING).readlines()
        else:
            lines = text

        char2idx, _ = create_vocab()
        input_lines = [normalize_text(line.strip()) + "$" for line in lines] # text normalization, $: EOS
        lengths = [len(line_in) for line_in in input_lines]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(input_lines), maxlen), np.int32)

        # Convert to int
        for i, line in enumerate(input_lines):
            texts[i, :len(line)] = [char2idx[char] for char in line]

        self.text = texts

if __name__ == "__main__":
    synth = Synthesizer()
    synth.synthesize(os.path.join(LOG_DIR, 'model-3250.meta'), text=['penguins. penguins everywhere',
                     'what the heck am i doing with my life',
                     'all around me are familiar faces',
                     'worn out faces',
                     'worn out places',
                     'is this the real life',
                     'is this just fantasy',
                     'caught in a landslide',
                     'no escape from reality'])
    print('done')