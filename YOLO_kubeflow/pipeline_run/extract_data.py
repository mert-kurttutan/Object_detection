import os

# Generic python packages
import numpy as np
import json
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import seaborn as sns
import random

# TF-related packages
import tensorflow as tf
from tensorflow import keras

# local packages
from src import (dataParser as dp, 
                 utils)


def extract_data():

    # PRECISION to be used for weights of model
    PRECISION = tf.float32


    # VOC
    os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
    os.system("tar xvf ./VOCtrainval_11-May-2012.tar")


