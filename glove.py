# Libraries are imported
import numpy as np
import os
import warnings
import pandas as pd
# Glove embedding matrix is extracted to the RAM(Carefull, needs lots of memory)
with open(os.getcwd() + '/glove.6B.300d.txt', 'r', encoding="utf8") as f:
    words = [[str(s) for s in line.rstrip().split(' ')] for line in f]
embedding_mat = [word[0] for word in words]

# Initialization of embedding matrix that will be used after this process
last_matrix = np.random.normal(0, 0.01, size=(1004, 300))

# Word Codes are extracted to a dictionary as we also did in main.py
warnings.filterwarnings("ignore")
word_codes = pd.read_hdf("eee443_project_dataset_train.h5", 'word_code')
cols = list(word_codes.columns)
word_nums = word_codes.values
dictionary = {}
for i, w in enumerate(cols):
    if (w == 'xCatch'):
        dictionary[word_nums[0][i]] = 'catch'
    elif (w == 'xWhile'):
        dictionary[word_nums[0][i]] = 'while'
    elif (w == 'xCase'):
        dictionary[word_nums[0][i]] = 'case'
    elif (w == 'xEnd'):
        dictionary[word_nums[0][i]] = 'end'
    elif (w == 'xFor'):
        dictionary[word_nums[0][i]] = 'for'
    else:
        dictionary[word_nums[0][i]] = w

# Vectors that corresponds word codes in our dataset is written into the last_matrix
for i in range(1000):
    print(i)
    last_matrix[i+4] = np.array([float(vec) for vec in words[embedding_mat.index(dictionary[i+4])][1:]])

# Last matrix is saved as numpy array
np.save(file="embedding_matrix.npy", arr=last_matrix)




















