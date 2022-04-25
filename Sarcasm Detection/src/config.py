# required libraries
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# hyperparameters
vocab_size = 10000
max_length = 32
embedding_dim = 16
padding_type='post'
oov_token = '<OOV>'