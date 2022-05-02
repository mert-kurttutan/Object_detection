import tensorflow as tf
import os

# Name of the labels in VOC
LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
              'bus',        'car',      'cat',  'chair',     'cow',
              'diningtable','dog',    'horse',  'motorbike', 'person',
              'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']
 
config = {
    "IMAGE_H"         : 416, 
    "IMAGE_W"        : 416,
    'GRID_H'          : 13,  
    'GRID_W'          : 13,
    'BOX'             : 4,
    'CLASS'          : LABELS,
    'batch_size'      : 16,
}


anchors_tf = tf.constant([0.08285376, 0.13705531,
                       0.20850361, 0.39420716,
                       0.80552421, 0.77665105,
                       0.42194719, 0.62385487])*config["GRID_W"]



TRAIN_STEPS = 100
EVAL_STEPS = 100


# directory of the raw data files
DATA_ROOT = './data/VOC_data'

#  Name of the pipeline to be passed pipeline constructor
PIPELINE_NAME = 'obj_det_pipe'

# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = './pipeline/'

# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata.db')

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = './serving_model'

# Path file of trainer module
TRANSFORM_MODULE = "./pipeline_run/obj_det_transform.py"

# Path file of trainer module
TRAINER_MODULE = "./pipeline_run/obj_det_trainer.py"
