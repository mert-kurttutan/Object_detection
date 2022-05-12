
import json
import random
import numpy as np
import os

# TF-related packages
import tensorflow as tf
from tensorflow import keras

# local packages
from src import (dataParser as dp, 
                 dataLoader as dl,
                 obj_det_model,
                 utils)

# PRECISION to be used for weights of model
PRECISION = tf.float32

def find_device(prior="CPU"):
    for my_device in tf.config.list_logical_devices():
        if my_device.device_type == prior:
            return my_device.name
        
        
        
# Device on which training to be done
DEVICE = find_device()



def transform_data():
    
    
    # Parse annotations 
    train_image_folder = "VOCdevkit/VOC2012/JPEGImages/"
    train_annot_folder = "VOCdevkit/VOC2012/Annotations/"


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
    
    with open("/home/jupyter/Object_detection/YOLO_kubeflow/train_image.json", "w") as f:
        f.write(jsonString)
        
    jsonString = json.dumps(LABELS)
    with open("/home/jupyter/Object_detection/YOLO_kubeflow/LABELS.json", "w") as f2:
        f2.write(jsonString)

