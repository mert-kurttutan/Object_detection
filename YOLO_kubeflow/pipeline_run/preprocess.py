# Script for preprocessing jobs in ML pipeline
import json
import random
import numpy as np
import os

# TF-related packages
import tensorflow as tf
from tensorflow import keras

# local packages
from src import (dataParser as dp, 
                 utils)

# TODO: Generalize to include options for different dataset
# TODO: Write input from pipeline to include hyperparameter settings


def preprocess(bucket: str, data_zip: str):
    
    # VOC
    os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
    os.system("tar xvf ./VOCtrainval_11-May-2012.tar")
    
    
    # Parse annotations 
    train_image_folder = "./Images/"
    train_annot_folder = "./VOCdevkit/VOC2012/Annotations/"


    # Name of the labels in VOC
    LABELS_VOC = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
                  'bus',        'car',      'cat',  'chair',     'cow',
                  'diningtable','dog',    'horse',  'motorbike', 'person',
                  'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

    for idx, label in enumerate(LABELS_VOC):
        LABELS_VOC[idx] = dp.cat_name_COCO_to_VOC(label)

    train_image_voc = dp.parse_annotation_VOC(train_annot_folder,train_image_folder, labels=LABELS_VOC)

    train_image = train_image_voc
    LABELS = LABELS_VOC

    
    jsonString = json.dumps(train_image)
    
    with open("./train_image.json", "w") as f:
        f.write(jsonString)
        
    jsonString = json.dumps(LABELS)
    with open("./LABELS.json", "w") as f2:
        f2.write(jsonString)
        
        
    
    # Get images directory to top level
    os.system("mv ./VOCdevkit/VOC2012/JPEGImages ./Images")
    
    # Delete unnecessay folders/files
    os.system("rm -rf ./VOCdevkit")
    
    # Compress training data
    os.system(f"zip -r {data_zip} Images LABELS.json train_image.json")
    
    # Upload GCS
    os.system(f"gsutil -m cp -r {data_zip} gs://{bucket}/")
    

