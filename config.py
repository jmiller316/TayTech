"""
config.py

Several constants used in the project
"""
#EPOCHS = 0
#BATCH_SIZE = 1
#RNN_SIZE = 0
ENCODING = 'UTF-8'
# JARED fill this in with the characters you are using and the 
# number of characters 
VOCAB_SIZE = 0
# VOCAB = ...
EMBED_SIZE = 256
NUM_HIGHWAY_LAYERS = 4
DROPOUT_RATE = 0.5
REDUCTION_FACTOR = 2 # This value can be changed. In the Tacotron paper, 
                     # the number 2 was used. The paper said, however,
                     # that numbers as large as 5 worked well
# JARED what is n_mels?
N_MELS = 80 # Number of Mel banks to generate