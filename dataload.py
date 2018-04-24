# -*- coding: utf-8 -*-
"""
Load the data for the model
"""
import numpy as np
import tensorflow as tf
from utils import create_vocab, normalize_text, create_spectrograms
import codecs
import os
from config import N_MELS, REDUCTION_FACTOR, N_FFT, BATCH_SIZE, DATA_PATH, DEVICE, ENCODING, NUM_EPOCHS

def input_load(mode="train"):
    """ Grab the text files and wav file names created by Garrett, Colin, and David and return them"""
    # creates vocab conversion dictionaries
    char2idx, _ = create_vocab()
    fpaths, text_lengths, texts = [], [], []
    
    base_path = os.path.join(DATA_PATH, 'data')
    base_path_g = os.path.join(base_path, 'Garrett')
    base_path_c = os.path.join(base_path, 'Colin')
    base_path_d = os.path.join(base_path, 'David')
    base_path_x = os.path.join(base_path, 'wavs')
    transcript_g = os.path.join(base_path_g, 'AudioTranscript.txt')
    transcript_c = os.path.join(base_path_c, 'Colin Freeman Audio Text.txt')
    transcript_d = os.path.join(base_path_d, 'transcript.csv')
    transcript_x = os.path.join(base_path_x, 'metadata.csv')
    
    if mode in ("train", "eval"):
        # Each epoch
        for _ in range(NUM_EPOCHS):
            #Comment this out then uncomment the other line loops to try morgan freeman
            lines = codecs.open(transcript_x, 'r', ENCODING).readlines()
            for line in lines:
                fname, _, text = line.strip().split("|")
    
                fpath = os.path.join(base_path_x, fname + ".wav")
                fpaths.append(fpath)
    
                text = normalize_text(text) + "$"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
            
        """ #Commented out to use a different dataset
        # Garrett
        lines = codecs.open(transcript_g, 'r', ENCODING).readlines()
        line_number = 0
        for line in lines:
            if line == "" or line == "\n":
                continue
            line_number += 1
            text, fname = line.strip().split("$")
            fname = fname.strip()
            if not fname:
                fpath = os.path.join(base_path_g,
                                     "G" + line_number + ".wav")
            else:
                fpath = os.path.join(base_path_g, fname)

            fpaths.append(fpath)                                        # queue of fpaths containing all the wavfiles

            text = normalize_text(text) + "$"  # $: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))

            texts.append(np.array(text, np.int32).tostring())

        # Colin
        lines = codecs.open(transcript_c, 'r', ENCODING).readlines()
        line_number = 0
        for line in lines:
            if line == "" or line == "\n":
                continue
            line_number += 1
            text, fname = line.strip().split("$")
            fname = fname.strip()
            if not fname:
                fpath = os.path.join(base_path_c,"C" + line_number + ".wav")
            else:
                fpath = os.path.join(base_path_c, fname)

            fpaths.append(fpath)                                        #queue of fpaths containing all the wavfiles

            text = normalize_text(text) + "$"  # E: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())           #queue of converted transcript text lines
        """
        
    """ DON'T Uncomment this doesn't access anything
    lines = codecs.open(transcript_d, 'r', 'utf-8').readlines()
    line_number = 0
    for line in lines:
        line_number += 1
        text, fname = line.strip().split("$")
        fname = fname.strip()
        if not fname: 
            fpath = os.path.join(base_path_d,"D" + line_number + ".wav")
        else:
            fpath = os.path.join(base_path_d, fname + ".wav")
            
        fpaths.append(fpath)                                        #queue of fpaths containing all the wavfiles

        text = normalize_text(text) + "$"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.int32).tostring())           #queue of converted transcript text lines
        """
        return fpaths, text_lengths, texts
    else:	+        fname = fname.strip()
-        # Parse	+        if not fname: 
-        lines = codecs.open(TEST_DATA, 'r', 'utf-8').readlines()[1:]	+            fpath = os.path.join(base_path_d,"D" + line_number + ".wav")
-        sents = [normalize_text(line.split(" ", 1)[-1]).strip() + "E" for line in	+        else:
-                    lines]  # text normalization, E: EOS	+            fpath = os.path.join(base_path_d, fname + ".wav")
-        lengths = [len(sent) for sent in sents]	+            
-        maxlen = sorted(lengths, reverse=True)[0]	+        fpaths.append(fpath)                                        #queue of fpaths containing all the wavfiles
-        texts = np.zeros((len(sents), maxlen), np.int32)	+
-        for i, sent in enumerate(sents):	+        text = normalize_text(text) + "$"  # E: EOS
-            texts[i, :len(sent)] = [char2idx[char] for char in sent]	+        text = [char2idx[char] for char in text]
-        return texts


def get_batch():
    """ Generate the batches """
    with tf.device(DEVICE): # You should set the DEVICE in the config file

        # load the data
        fpaths, text_lengths, texts = input_load()
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # determine number of batches
        num_batch = len(fpaths) // BATCH_SIZE

        # Tensors
        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True) #forms queues from lists

        # Set i[ fpr creatomg s[ectpgra,s
        fname, mel, mag = tf.py_func(create_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        # Set shape
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, N_MELS*REDUCTION_FACTOR))
        mag.set_shape((None, N_FFT//2+1))

        # Get buckets
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=BATCH_SIZE,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=16,
                                            capacity= BATCH_SIZE * 4,
                                            dynamic_pad=True)

        return texts, mels, mags, fnames, num_batch

