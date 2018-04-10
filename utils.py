"""
utils.py

Miscellaneous functions that we will need.
"""
import librosa
import numpy as np
from scipy import signal
import copy
from config import *

"""
attention
"""

"""
pre-net
"""

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
    
    y, sample_rate = librosa.load(fpath, sample_rate=sample_rate)                   #load the wav file
    
    y, _ = librosa.effects.trim(y)                          #trimming
    
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])    #noise reduction technique
    
    linear = librosa.stft(y=y,                              #short time fourier transform to get the linear spectrogram
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)
    
    mag = np.abs(linear)
    
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    
    return mel,mag

def spectrogram2wav(spectrogram):
    '''# Generate wave file from spectrogram'''
    # transpose
    spectrogram = spectrogram.T

    # de-noramlize
    spectrogram = (np.clip(spectrogram, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    spectrogram = np.power(10.0, spectrogram * 0.05)

    # wav reconstruction
    wav = griffin_lim(spectrogram)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
    y = np.real(X_t)
    
    return y

def create_spectrograms():
    fname = os.path.basename(fpath)
    mel, mag = wav2spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = 3 - (t % 3) if t % 3 != 0 else 0 # for reduction using 3 as reduction factor
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    
    return fname, mel.reshape((-1, n_mels*3)), mag