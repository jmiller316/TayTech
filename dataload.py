# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:18:54 2018

@author: jared
"""
import numpy as np
import tensorflow as tf
from utils import create_vocab, normalize_text, create_spectrograms
import codecs
import os
from config import *

def input_load():
    #creates vocab conversion dictionaries
    char2idx, _ = create_vocab()
    fpaths, text_lengths, texts = [], [], []
    
    transcript = os.path.join("/data/Garret", 'transcript.csv')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    line_number = 0
    for line in lines:
        line_number += 1
        text, fname = line.strip().split("$")
        if not fname:
            fpath = os.path.join("/data/Garret","G" + line_number + ".wav")
        else:
            fpath = os.path.join("/data/Garret",fname + ".wav")
            
        fpaths.append(fpath)                                        #queue of fpaths containing all the wavfiles

        text = normalize_text(text) + "$"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.int32).tostring())           #queue of converted transcript text lines
        
    transcript = os.path.join("/data/Collin", 'transcript.csv')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    line_number = 0
    for line in lines:
        line_number += 1
        text, fname = line.strip().split("$")
        if not fname:
            fpath = os.path.join("/data/Collin","C" + line_number + ".wav")
        else:
            fpath = os.path.join("/data/Collin", fname + ".wav")
            
        fpaths.append(fpath)                                        #queue of fpaths containing all the wavfiles

        text = normalize_text(text) + "$"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.int32).tostring())           #queue of converted transcript text lines
        
    transcript = os.path.join("/data/David", 'transcript.csv')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    line_number = 0
    for line in lines:
        line_number += 1
        text, fname = line.strip().split("$")
        if not fname:
            fpath = os.path.join("/data/David","D" + line_number + ".wav")
        else:
            fpath = os.path.join("/data/David", fname + ".wav")
            
        fpaths.append(fpath)                                        #queue of fpaths containing all the wavfiles

        text = normalize_text(text) + "$"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.int32).tostring())           #queue of converted transcript text lines
        
    return fpaths, text_lengths, texts

def get_batch():
    with tf.device('/device:GPU:0'): #this uses your primary gpu use ('/cpu:0') to use your cpu instead
        
        fpaths, text_lengths, texts = input_load()
        maxlen, minlen = max(text_lengths), min(text_lengths)
        
        num_batch = len(fpaths)
        
        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)
        
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True) #forms queues from lists
        
        fname, mel, mag = tf.py_func(create_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, N_MELS*REDUCTION_FACTOR))
        mag.set_shape((None, N_FFT//2+1))
        
        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)
        
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=16,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=16,
                                            capacity= BATCH_SIZE * 4,
                                            dynamic_pad=True)
        
        return texts, mels, mags, fnames, num_batch
