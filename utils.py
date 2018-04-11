"""
utils.py

Miscellaneous functions that we will need.
"""
import tensorflow as tf
import librosa
import numpy as np
import os
from scipy import signal
import copy
from config import MAX_DB,REF_DB, WIN_LENGTH, \
    N_FFT, PREEMPHASIS, HOP_LENGTH, N_MELS, N_ITER, VOCAB, SR

def attention(inputs, memory, num_units=None, scope="attention_decoder"):
    with tf.variable_scope(scope):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, 
                                                                   memory),
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                                  attention_mechanism,
                                                                  num_units,
                                                                  alignment_history=True)
        outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32) #( N, T', 16)

    return outputs, state

def learning_rate_decay(global_step, warmup_steps=4000.0):
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

"""
text input edit
"""
def create_vocab():
    char2idx = {char: idx for idx, char in enumerate(VOCAB)}
    idx2char = {idx: char for idx, char in enumerate(VOCAB)}
    return char2idx,idx2char

def normalize_text(text):
    #room for more normalization depending on transcript data
    text = text.lower().replace(",",".")
    return text
"""
audio edit
"""
def wav2spectrograms(fpath):
    
    y, sample_rate = librosa.load(fpath, sr=SR)                   #load the wav file
    
    y, _ = librosa.effects.trim(y)                          #trimming
    
    y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])    #noise reduction technique
    
    linear = librosa.stft(y=y,                              #short time fourier transform to get the linear spectrogram
                          n_fft=N_FFT,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH)
    
    mag = np.abs(linear)
    
    mel_basis = librosa.filters.mel(sample_rate, N_FFT, N_MELS)  # (n_mels, 1+n_fft//2)
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
    '''# Generate wave file from spectrogram'''
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

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
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
    num_paddings = 3 - (t % 3) if t % 3 != 0 else 0 # for reduction using 3 as reduction factor
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    
    return fname, mel.reshape((-1, N_MELS*3)), mag