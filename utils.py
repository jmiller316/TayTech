"""
utils.py

Miscellaneous functions that we will need.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
from scipy import signal
import copy
from scipy.io import wavfile
from config import MAX_DB,REF_DB, WIN_LENGTH, \
    N_FFT, PREEMPHASIS, HOP_LENGTH, N_MELS, N_ITER, VOCAB, SR, LOG_DIR, CHECK_VALS, \
    REDUCTION_FACTOR


def attention(inputs, memory, num_units=None, scope="attention_decoder"):
    with tf.variable_scope(scope):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory)
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                                  attention_mechanism,
                                                                  num_units,
                                                                  alignment_history=True)
        outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32) #( N, T', 16)

    return outputs, state


def learning_rate_decay(global_step):
    """
    Learning_rate_decay.

    The learning rate decay starts from 0.001 and is reduced to 0.0005, 0.0003, 
    and 0.0001 after 500K, 1M, and 2M global steps respectively.

    While the rate is hardcoded in here, it can be changed to the Naome from 
    tensor2tensor if it does not work well. 
    """
    # Constants for the learning rate
    r1 = tf.constant(0.001)
    r2 = tf.constant(0.0005)
    r3 = tf.constant(0.0003)
    r4 = tf.constant(0.0001)
    l1 = tf.constant(500000)
    l2 = tf.constant(1000000)
    l3 = tf.constant(2000000)

    def ifr1():
        return r1

    def if2():
        return tf.cond(tf.less(global_step, l2), ifr2, if3)

    def ifr2():
        return r2

    def if3():
        return tf.cond(tf.less(global_step, l3), ifr3, ifr4)

    def ifr3():
        return r3

    def ifr4():
        return r4

    return tf.cond(tf.less(global_step, l1),ifr1,if2)

def plot_alignments(alignment, global_step):
    """
    Plots the alignments
    """
    plt.plot()
    ax = plt.subplot()
    im = ax.imshow(alignment)

    plt.colorbar(im, ax=ax)
    plt.title(str(global_step) + " global steps")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'alignment_' + str(global_step//CHECK_VALS) + '_k.png'), format='png')


def create_vocab():
    """
    text inpu
    char2idx = {char: idx for idx, char in enumerate(VOCAB)}
    idx2char = {idx: char for idx, char in enumerate(VOCAB)}
    return char2idx,idx2char


def normalize_text(text):
    # room for more normalization depending on transcript data
    text = text.lower().replace(",",".")
    text = text.replace("-"," ")
    return text


def wav2spectrograms(fpath):
    """
    audio edit
    """
    y, sample_rate = librosa.load(fpath, sr=SR)                   # load the wav file
    y, _ = librosa.effects.trim(y)                          # trimming
    
    y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])    # noise reduction technique
    
    linear = librosa.stft(y=y,                              # short time fourier transform to get the linear spectrogram
                          n_fft=N_FFT,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH)
    
    mag = np.abs(linear)
    
    mel_basis = librosa.filters.mel(SR, N_FFT, N_MELS)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)
    mag = np.clip((mag - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    
    return mel,mag


def spectrogram2wav(spectrogram):
    """# Generate wave file from spectrogram"""
    # transpose
    spectrogram = spectrogram.T

    # de-normalize
    spectrogram = (np.clip(spectrogram, 0, 1) * MAX_DB) - MAX_DB + REF_DB

    # to amplitude
    spectrogram = np.power(10.0, spectrogram * 0.05)

    # wav reconstruction
    wav = griffin_lim(spectrogram)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -PREEMPHASIS], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

    return y

def griffin_lim(spectrogram):
    """Applies Griffin-Lim's raw."""
    X_best = copy.deepcopy(spectrogram)
    for i in range(N_ITER):
        X_t = librosa.istft(X_best, HOP_LENGTH, win_length=WIN_LENGTH, window="hann")
        est = librosa.stft(X_t, N_FFT, HOP_LENGTH, win_length=WIN_LENGTH)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, HOP_LENGTH, win_length=WIN_LENGTH, window="hann")
    y = np.real(X_t)
    
    return y

def create_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = wav2spectrograms(fpath)

    t = mel.shape[0]

    num_paddings = REDUCTION_FACTOR - (t % REDUCTION_FACTOR) \
        if t % REDUCTION_FACTOR != 0 else 0

    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    
    return fname, mel.reshape((-1, N_MELS*REDUCTION_FACTOR)), mag

def wav2drawing(fpath):
    sample_rate, samples = wavfile.read(fpath)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

if __name__ == "__main__":
    path = 'C:\\Users\\Sabrina\\Documents\\UTSA\\Intro to AI\\Group Project\\data\\Garrett\\g24.wav'
    _, mag = wav2spectrograms(path)
    plot_alignments(mag, global_step=1000)