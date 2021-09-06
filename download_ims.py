# Libraries are imported
import requests
import time
import concurrent.futures
import os
import h5py
import numpy as np
from skimage import io
from skimage.transform import resize

# Read provided dataset
train = h5py.File('eee443_project_dataset_train.h5', 'r')
test = h5py.File('eee443_project_dataset_test.h5', 'r')

# Urls of train and test parts are extracted from dataset
train_url = np.array(train['train_url'])
print("train_url Shape:", train_url.shape)
test_url = np.array(test['test_url'])
print("test_url Shape:", test_url.shape)


# Function downloads image into the dataset(Only test dataset is written down here)
def download_image(img_url, count):
    try:
        im = io.imread(img_url)
        resized = resize(im, (224, 224), anti_aliasing=True)
        io.imsave(fname=os.getcwd() + "/test_images/" + str(count) + ".jpg", arr=(resized * 255).astype(np.uint8))
    except:
        pass

# Multithread download of images whose urls are given
count = 0
with concurrent.futures.ThreadPoolExecutor() as executor:
    for url in test_url:
        executor.submit(download_image, url.decode(), count)
        count += 1

