"""
config.py

Several constants used in the project
"""
ENCODING = 'UTF-8'
NUM_EPOCHS = 20
# JARED fill this in with the characters you are using and the 
# number of characters 
VOCAB_SIZE = 32
VOCAB = "_$ abcdefghijklmnopqrstuvwxyz'.?"  # _ = padding, $ = ending

EMBED_SIZE = 256
NUM_HIGHWAY_LAYERS = 4
DROPOUT_RATE = 0.5
REDUCTION_FACTOR = 2 # This value can be changed. In the Tacotron paper, 
                     # the number 2 was used. The paper said, however,
                     # that numbers as large as 5 worked well
LOG_DIR = 'C:\\Users\\Sabrina\\Documents\\UTSA\\Intro to AI\\Group Project\\data_log'
DATA_PATH = 'C:\\Users\\Sabrina\\Documents\\UTSA\\Intro to AI\\Group Project'
DEVICE = '/cpu:0'
# Signal Processing
# JARED we need to look at these values since they depend on the data
SR = 22050 # Sample rate.
# 80 band mel scale spetrogram
N_MELS = 80
N_FFT = 2048 # fft points (samples)
MAX_DB = 100
REF_DB = 20
FRAME_SHIFT = 0.0125 # seconds
FRAME_LENGTH = 0.05 # seconds
HOP_LENGTH = int(SR*FRAME_SHIFT) # samples.
WIN_LENGTH = int(SR*FRAME_LENGTH) # samples.
POWER = 1.2 # Exponent for amplifying the predicted magnitude
N_ITER = 50 # Number of inversion iterations
PREEMPHASIS = .97 # or None
BATCH_SIZE = 20
CHECK_VALS = 5
