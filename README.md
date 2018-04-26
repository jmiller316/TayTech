# TayTech



## Installation Instructions. 
1. Download the code. Download the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Modify config.py. 
  a. LOG_DIR: location of model checkpoints
  b. MODEL_NAME: the name of the model file to load
  c. DATA_PATH: location of the dataset
  d. TEST_DATA: text file used to synthesize audio if not passing in text to predict in the code
  e. SAVE_DIR: where to save sythesized outputs
  f. DEVICE: cpu or gpu depending on your computer
 
## Instruction to Run
1. Training - run train.py to train
2. Evaluation - run eval.py to evaluate
3. Synthesize - run synthesizer.py to synthesize output that is stored in SAVE_DIR. Either pass in text in the code or modify TEST_DATA

## Results

## Dependencies
Runs on Python 3
- Conda >= 4.4.10
- tensorflow >= 1.6
- librosa >= 0.6.0
- numpy >= 1.14.1
- matplotlib >= 2.2.2
- scipy >= 1.0.0
- tqdm

## Data
Two datasets were used. 
  - [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
  - [Morgan Freeman Audio](https://drive.google.com/drive/folders/1efzGhWzDOpSxnCnofrmSYX_j7cYokCzN)
  The model was trained on the LJ Speech Dataset. Although the Morgan Freeman Data was gathered, it was insufficient for training the model.

## References
- Tacotron: https://arxiv.org/abs/1703.10135
- Highway networks: https://arxiv.org/pdf/1505.00387.pdf
- LJ Speech Dataset: https://keithito.com/LJ-Speech-Dataset/
- Morgan Freeman Audio: https://youtube.com/
- Github repos: 
  - https://github.com/Kyubyong/tacotron
  - https://github.com/keithito/tacotron
