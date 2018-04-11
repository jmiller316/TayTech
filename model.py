"""
Contains the class for the main model
"""

from dataload import get_batch
import tensorflow as tf
from networks import encoder, decoder
from utils import spectrogram2wav, learning_rate_decay
from config import N_MELS, REDUCTION_FACTOR, N_FFT, EMBED_SIZE, SR
from cbhg import cbhg_helper

class Model:
    """
    Generate the Tensorflow model
    """
    def __init__(self, mode="train"):
        self.mode = mode

        # If is_training
        if mode=="train":
            self.is_training = True
        else:
            self.is_trainig = False

        # Load inputs
        if self.is_training:
            self.txt, self.mels, self.mags, self.file_names, self.num_batch = get_batch()
        else:
            self.txt = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, N_MELS*REDUCTION_FACTOR))
            self.mags = tf.placeholder(tf.float32, shape=(None, None, 1+ N_FFT//2))
            self.file_names = tf.placeholder(tf.string, shape=(None,))      

        # decoder inputs
        self.decoder_inputs = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1) # (N, Ty/r, n_mels*r)
        self.decoder_inputs = self.decoder_inputs[:, :, -N_MELS:] # feed last frames only (N, Ty/r, n_mels)

        # Networks
        with tf.variable_scope("Networks"):
            # encoder
            self.memory = encoder(self.txt, is_training=self.is_training)

            # decoder
            self.mel_hat, self.alignments = decoder(self.decoder_inputs, self.memory, is_training=self.is_training)

            # CBHG Module
            self.mags_hat = cbhg_helper(self.mel_hat, EMBED_SIZE//2, is_training=self.is_training)

        # audio
        self.audio_out = tf.py_func(spectrogram2wav, [self.mags_hat[0]], tf.float32)

        # Training and evaluation
        if mode in ("train", "eval"):
            # Loss
            self.loss = loss()

            # Training Scheme
            self.optimizer = optimize()

            """
            Not part of the reasearch paper
            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
            """ 
            summarize()

        def loss(self):
            """
            Determine the loss of the outputs
            """
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_hat - self.mels))
            self.mag_loss = tf.reduce_mean(tf.abs(self.mags_hat - self.mags))
            return self.mel_loss + self.mag_loss

        def optimize(self):
            """
            Optimize the learning rate.
            """
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learn_rate = learning_rate_decay(global_step=self.global_step)
            return tf.train.AdamOptimizer(learning_rate=self.learn_rate)

        def summarize(self):
            """
            Summarize the training
            """
            tf.summary.scalar('mode: %s \nmel loss: ' %(mode), self.mel_loss)
            tf.summary.scalar('mag loss:', self.mag_loss)
            tf.summary.scalar('total loss:', self.loss)
            tf.summary.scalar('learning rate: ', self.learning_rate)
            tf.summary.image('Mel input:', tf.expand_dims(self.mels, -1), max_outputs=1)
            tf.summary.image('Mel output:', tf.expand_dims(self.mel_hat, -1), max_outputs=1)
            tf.summary.image('Mag input:', tf.expand_dims(self.mags, -1), max_outputs=1)
            tf.summary.image('Mag output:', tf.expand_dims(self.mags_hat, -1), max_outputs=1)
            tf.summary.audio('Audio:', tf.expand_dims(self.audio_out, 0), SR)
            self.merged = tf.summary.merge_all()