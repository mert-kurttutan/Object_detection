import json
import random
import numpy as np
import os
from google.cloud import storage

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
    return my_device.name
        
        
        
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f'Blob {source_blob_name} downloaded to {destination_file_name}.')

        



def train(bucket, data_zip, processor="CPU"):
    
    # Device on which training to be done
    DEVICE = find_device(prior=processor)

    _ANCHORS01 = np.array([0.19, 0.40,
                           0.79, 0.74,
                           0.086, 0.12,
                           0.41, 0.59])

    GRID_H,  GRID_W  = 13 , 13
    ANCHORS          = _ANCHORS01.astype(np.single)
    ANCHORS[::2]     = ANCHORS[::2]*GRID_W  
    ANCHORS[1::2]    = ANCHORS[1::2]*GRID_H  
    
    anchors_tf = tf.convert_to_tensor(ANCHORS)
    
    # Download zip file for preprocessed training data
    
    download_blob(bucket, data_zip, data_zip)
    os.system(f"unzip {data_zip}")
    train_image = json.load(open("./train_image.json", 'r'))
    LABELS = json.load(open("./LABELS.json", 'r'))

    IMAGE_H, IMAGE_W = 416, 416
    BATCH_SIZE       = 16
    BOX = int(len(ANCHORS)/2)
    CLASS = len(LABELS)

    obj_det_config = {
        'image_h'         : IMAGE_H, 
        'image_w'         : IMAGE_W,
        'grid_h'          : GRID_H,  
        'grid_w'          : GRID_W,
        'box'             : BOX,
        'classes'          : LABELS,
        'anchors'         : ANCHORS,
        'batch_size'      : BATCH_SIZE,
    }

    shuffle = True
    
    
    random.seed(43)
    # Shuffle the data before loading
    random.shuffle(train_image)
    split = int(len(train_image) * 0.85)

    train_batch_generator = dl.YOLOData(train_image[:split], obj_det_config,
                                            norm=True, shuffle=shuffle)

    val_batch_generator = dl.YOLOData(train_image[split:], obj_det_config,
                                            norm=True, shuffle=False)
    
    myVGG = obj_det_model.VGGYOLO(CLASS=len(LABELS), BOX=len(_ANCHORS01)//2)
    
    
    # Use pretrained weights from original YOLOV2 paper
    os.system("wget https://pjreddie.com/media/files/yolov2.weights")
    
    path_to_weight = "./yolov2.weights"
    weight_reader = obj_det_model.WeightReaderDarkNet19(path_to_weight)
    
    
    weight_reader.reset()
    nb_conv = 23
    
    # Load the weights to the model
    for i in range(1, nb_conv):
        conv_layer = myVGG.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = myVGG.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])       

        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])



    for i in range(2):  
        layer   = myVGG.layers[-(i+2)] # the last convolutional layer
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
        new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

        layer.set_weights([new_kernel, new_bias])
        
    
    print("Finished setting the weights of the model.......")
    
    loss = obj_det_model.YoloLoss(anchors_tf, len(LABELS),
              w_coord=.5, w_class=1, w_conf=.5, lam_obj=5, lam_noobj=1,
              PRECISION = tf.float32, print_loss = False) 
    
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop

    dir_log = "logs/"
    try:
        os.makedirs(dir_log)
    except:
        pass


    obj_det_config['BATCH_SIZE'] = BATCH_SIZE

    early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=0.0001, 
                               patience=2, 
                               mode='min', 
                               verbose=1)

    checkpoint = ModelCheckpoint('./models/yolo_v2/weights_v1.h5', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 save_freq="epoch")
    with tf.device(DEVICE):
        optimizer = Adam(learning_rate=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        myVGG.compile(loss=loss ,optimizer=optimizer)
        
        
    print("Started training ......")
    myVGG.fit(  x = train_batch_generator, 
                validation_data = val_batch_generator,
                steps_per_epoch  = len(train_batch_generator), 
                epochs           = 100, 
                verbose          = 1,
                callbacks        = [early_stop, ], 
                max_queue_size   = 3)