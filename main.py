import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

import cv2
import os

import numpy as np
import pandas as pd
import math
import random

from batchFunctions import *

# DATA DEFINITIONS
path = '../input/petfinder-pawpularity-score/'
scores = pd.read_csv(path + 'train.csv')
img_size=300

# BATCH DEFINITIONS
batches = 21
batch_length = math.floor(scores.shape[0] / batches)
print(f'Total Length: {scores.shape[0]} | Batches: {batches} | Batch Length: {batch_length}')

# CREATE MODEL
from wrapper import wrapper

# INITIALIZE TEST DATA
test_data = []
test_results = []
test_ids = []

# TEST ALL IMAGES IN FOLDER
for filename in os.listdir(path + 'test'):
    try:
        test_ids.append(filename[:-4])
        img_arr = cv2.imread(os.path.join(path + 'test', filename))[...,::-1] 
        resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
        test_data.append(resized_arr)
        test_data = np.array(test_data) / 255
        test_data = preprocessdata(test_data)
        test_results.append(wrapper.predict(test_data).tolist()[0])
        test_data = []
        
    except Exception as e:
        print(e)

# SET UP OUTPUT FILE  
output_dict = {'Id': test_ids, 'Pawpularity': test_results}
output_df = pd.DataFrame(output_dict)
output_df.to_csv("output.csv",index=False)