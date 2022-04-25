# required libraries
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# hyperparameters
vocab_size = 10000
max_length = 32
embedding_dim = 32
padding_type='post'
oov_token = '<OOV>'
number_of_epochs = 10
lstm1_dim = 64
lstm2_dim = 32
gru_dim = 32
filters = 128
kernel_size = 5
lr = 0.0001