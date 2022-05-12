import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import (Reshape, Activation, Conv2D, Input, MaxPooling2D,
                     BatchNormalization, Flatten, Dense, Lambda, LeakyReLU, concatenate)
from tensorflow.keras.layers import concatenate

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)


# Subclass version of convblock
class ConvBlock_obj(tf.keras.Model):
  def __init__(self, chann_in, chann_out, k_size, stride, pad_value, idx='', is_max=False):
    self.chann_in = chann_in
    self.chann_out = chann_out
    self.k_size = k_size
    self.stride = stride
    self.pad_value = pad_value
    self.idx = idx
    self.is_max = is_max
    
    self._build()

  def __call__(self, inputs):
    x = Conv2D(filters=self.chann_out, kernel_size=self.k_size, strides=self.stride, padding=self.pad_value)(inputs)
    x = BatchNormalization(f'norm_{self.idx}')(x)
    x = LeakyReLU(alpha=0.1)(x)

    if self.is_max:
      x = MaxPooling2D(pool_size=(2, 2))(x)

    return x
    
  def _build(self):
      
    inputs  = Input(shape=[16, 16, 12])
    outputs = self.__call__(inputs)
    super().__init__(name="ConvBlock", inputs=inputs, outputs=outputs) 


# Functional version of convblock
def conv_block_func(chann_in, chann_out, k_size, stride, pad_value, idx='', is_max=False):

  def conv_block(inputs):
  
    x = Conv2D(filters=chann_out, kernel_size=k_size, strides=stride, padding=pad_value, use_bias=False, name=f'conv_{idx}')(inputs)
    x = BatchNormalization(name=f'norm_{idx}')(x)
    x = LeakyReLU(alpha=0.1)(x)

    if is_max:
      x = MaxPooling2D(pool_size=(2, 2))(x)      

    return x
  
  return conv_block

def repeat_conv_block(x, param_arr, indices):
  '''
  Repeat the conv_block with list of parameters in indices
  '''


  for param, idx in zip(param_arr, indices):
    is_max, chann_in, chann_out, k_size, stride, pad_value = param
    conv_block = conv_block_func(chann_in, chann_out, k_size, stride, pad_value, idx, is_max)
    x = conv_block(x)

  return x


class VGGYOLO(tf.keras.Model):
  """TF Model class that creates VGGYOLO Model based on darknet19 architecture"""
  def __init__(self, inputs=None, CLASS=10, BOX=7,
                GRID_H=13, GRID_W=13,IMAGE_H=416, IMAGE_W=416):
    # Parametes for classification problem
    self.inputs = inputs
    self.CLASS = CLASS
    self.BOX = BOX
    self.GRID_H = GRID_H
    self.GRID_W = GRID_W
    self.IMAGE_H = IMAGE_H
    self.IMAGE_W = IMAGE_W

    # Arrays to provide indices of conv_block, see their names
    self.indices_1 = np.arange(1, 14, 1, dtype=np.int32)
    self.indices_2 = np.arange(14, 21, 1, dtype=np.int32)
    self.indices_skip = np.arange(21, 22, 1, dtype=np.int32)
    self.indices_3 = np.arange(22, 23, 1, dtype=np.int32)

    # Parameters of conv_blocks
    self.param_arr_1 = [
                    (True, 3, 32, 3, 1, 'same'),    # 1
                    (True, 32, 64, 3, 1, 'same'),    # 2
                    (False, 64, 128, 3, 1, 'same'),   # 3
                    (False, 128, 64, 1, 1, 'same'),    # 4
                    (True, 64, 128, 3, 1, 'same'),     # 5
                    (False, 128, 256, 3, 1, 'same'),    # 6
                    (False, 256, 128, 1, 1, 'same'),     # 7
                    (True, 128, 256, 3, 1, 'same'),     # 8
                    (False, 256, 512, 3, 1, 'same'),     # 9
                    (False, 512, 256, 1, 1, 'same'),     # 10
                    (False, 256, 512, 3, 1, 'same'),     # 11
                    (False, 512, 256, 1, 1, 'same'),     # 12
                    (False, 256, 512, 3, 1, 'same'),     # 13
                ]

    self.param_arr_2 = [
                    (False, 512, 1024, 3, 1, 'same'),    # 14
                    (False, 1024, 512, 1, 1, 'same'),    # 15
                    (False, 512, 1024, 3, 1, 'same'),    # 16
                    (False, 1024, 512, 1, 1, 'same'),    # 17
                    (False, 512, 1024, 3, 1, 'same'),    # 18
                    (False, 1024, 1024, 3, 1, 'same'),    # 19
                    (False, 1024, 1024, 3, 1, 'same'),     # 20
                ]

    self.param_arr_skip = [(False, 512, 64, 1, 1, 'same')]      # 21 dont forget to add space_depth
    self.param_arr_3 = [(False, 1280, 1024, 3, 1, 'same')]      # 22

    self._build()

    
  def _build(self):
      
    # Conv blocks (BatchNorm + ReLU activation added in each block)
    if self.inputs is None:
      self.inputs = Input(shape=(self.IMAGE_H, self.IMAGE_W, 3))
      x = self.inputs

    else:
      x = tf.keras.layers.concatenate(tf.nest.flatten(self.inputs))

    x = repeat_conv_block(x, self.param_arr_1, self.indices_1)
    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = repeat_conv_block(x, self.param_arr_2, self.indices_2)
    skip_connection = repeat_conv_block(skip_connection, self.param_arr_skip, self.indices_skip) 
    skip_connection = Lambda(space_to_depth_x2)(skip_connection) 

    x = concatenate([skip_connection, x])             
    x = repeat_conv_block(x, self.param_arr_3, self.indices_3)

    # Last fully connected layer to make prediction
    x = Conv2D(filters=1024,  # 23
                    kernel_size=1, padding='same', strides=(1,1), activation="relu", name="fc1")(x)
    x =  Conv2D(filters=self.BOX * (4 + 1 + self.CLASS),  # 24
                    kernel_size=1, padding='same', strides=(1,1), name="fc2")(x)

    x = Reshape((self.GRID_H, self.GRID_W, self.BOX, 4 + 1 + self.CLASS))(x)

    super().__init__(name="VGGYOLO", inputs=self.inputs, outputs=x) 

class WeightReaderDarkNet19:
  '''
  Reader object that reades numerical values from binary files to weights of darknet19 model
  The binary file to be read is https://pjreddie.com/media/files/yolov2.weights
  '''
  def __init__(self, weight_file):
    self.current_start = 4
    self.all_weights = np.fromfile(weight_file, dtype='float32')

  def read_bytes(self, size):
    self.current_start = self.current_start + size
    return self.all_weights[self.current_start-size:self.current_start]

  def reset(self):
    self.current_start = 4


def get_cell_grid(grid_w, grid_h, batch_size, box_num, device, PRECISION=tf.float32): 
    '''
    Helper function to assure that the bounding box x and y are in the grid cell scale
    == output == 
    for any i=0,1..,batch size - 1
    output[i,5,3,:,:] = array([[3., 5.],
                               [3., 5.],
                               [3., 5.]], dtype=float32)
    '''

    with tf.device(device):
      cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)), PRECISION)
      cell_y = tf.transpose(cell_x, (0,2,1,3,4))  
      cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, box_num, 1])
      
    return cell_grid 



def iou(pred_xy, pred_wh, 
        true_xy, true_wh):


  true_wh_half = true_wh / 2.
  true_mins    = true_xy - true_wh_half
  true_maxes   = true_xy + true_wh_half
  
  pred_wh_half = pred_wh / 2.
  pred_mins    = pred_xy - pred_wh_half
  pred_maxes   = pred_xy + pred_wh_half    
  
  intersect_mins  = tf.maximum(pred_mins,  true_mins)
  intersect_maxes = tf.minimum(pred_maxes, true_maxes)
  intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
  intersect_areas = intersect_wh[...,0] * intersect_wh[...,1]
  true_areas = true_wh[...,0] * true_wh[...,1]
  pred_areas = pred_wh[...,0] * pred_wh[...,1]

  union_areas = pred_areas + true_areas - intersect_areas
  iou_scores  = tf.truediv(intersect_areas, union_areas + 1e-4) 
  return iou_scores



def yolo_max_iou(pred_xy, pred_wh, true_boxes):

  pred_xy_c = tf.expand_dims(pred_xy, 4)
  pred_wh_c = tf.expand_dims(pred_wh, 4)

  true_xy = true_boxes[..., 0:2]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)
  true_wh = true_boxes[..., 2:4]           # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)

  iou_scores = iou(pred_xy_c, pred_wh_c, true_xy, true_wh)

  max_ious = tf.reduce_max(iou_scores, axis=4)
  
  return max_ious


def obj_mask_iou(obj_mask,
                pred_box_xy, pred_box_wh,
                true_box_xy,  true_box_wh):
    ''' 
    == input ==
    
    true_box_conf : tensor of shape (N_batch, N_anchor, N_grid_h, N grid_w )
    true_box_xy   : tensor of shape (N_batch, 2, N_anchor, N_grid_h, N grid_w)
    true_box_wh   : tensor of shape (N_batch, 2, N_anchor, N_grid_h, N grid_w)
    pred_box_xy   : tensor of shape (N_batch, 2, N_anchor, N_grid_h, N grid_w)
    pred_box_wh   : tensor of shape (N_batch, 2, N_anchor, N_grid_h, N grid_w)
        
    == output ==
    
    true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor)
    
    true_box_conf value depends on the predicted values 
    true_box_conf = IOU_{true,pred} if objecte exist in this anchor else 0
    '''
    iou_scores        =  iou(pred_box_xy,pred_box_wh,
                                    true_box_xy, true_box_wh)
    obj_mask_IOU = iou_scores * tf.squeeze(obj_mask)
    return obj_mask_IOU




def yolo_head(yolo_pred, anchors, grid=13, threshold=6, PRECISION=tf.float32):
  '''
  Returns the correct values for object detection problem, 
  given output of YOLOV2 Network

  Note: pred_class_prob is actually logit value for numerical efficiency, stability
  
  '''
  
  box_num = int(anchors.shape[0]/2)
  device = yolo_pred.device
  
  batch_size = tf.cast(tf.shape(yolo_pred)[0], tf.float32)
  cell_grid = get_cell_grid(grid,grid,batch_size,box_num,device,PRECISION)
  # Relative position of center within grid cell
  pred_box_xy = tf.sigmoid(yolo_pred[:,:,:,:,:2]) + cell_grid

  
  with tf.device(device):
    tns_exp = tf.exp(tf.constant([threshold/2]))-threshold

  # Relative width boxes w.r.t grid cell
  box_arr = tf.reshape(anchors,[1,1,1,box_num,2])
  # Rescale the prior boxes to predict shape of object
  # pred_box_wh = torch.exp(yolo_pred[:,2:4,:,:,:]) * box_arr
  

  # pred_box_wh = tf.exp(yolo_pred[:,:,:,:,2:4]) * box_arr # bw, bh
  mask = tf.cast(yolo_pred[:,:,:,:,2:4] > threshold, PRECISION)
  pred_box_wh = (tf.square(yolo_pred[:,:,:,:,2:4] + tns_exp) * mask + tf.exp(yolo_pred[:,:,:,:,2:4]*(1-mask))* (1-mask)) * box_arr
  
  # Confidence is a probability
  # pred_box_conf = torch.sigmoid(yolo_pred[:,4,:,:,:])
  # Confidence probabilities, actually logit values, to make it numerically stable
  pred_box_conf = yolo_pred[:,:,:,:,4]

  # Class probabilities, actually logit values, to make it numerically stable
  pred_class_prob = yolo_pred[:,:,:,:,5:]

  return pred_box_xy, pred_box_wh, pred_box_conf, pred_class_prob



def yolo_coord_loss(pred_box_xy, pred_box_wh, 
                    true_box_xy, true_box_wh, 
                    obj_mask):
  
  # Normalization so that when summing over 1 is equal to 1
  # Applied to the other terms in the loss function
  N_obj = tf.reduce_sum(tf.cast(obj_mask>0,dtype=tf.float32))

  yolo_xy_loss = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * obj_mask) / (N_obj+1e-4)
  yolo_wh_loss = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * obj_mask) / (N_obj+1e-4)

  return yolo_wh_loss + yolo_xy_loss




def yolo_class_loss(pred_box_class, true_box_class,
                    obj_mask):

  N_obj = tf.reduce_sum(tf.cast(obj_mask>0,dtype=tf.float32))
  loss_class   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_box_class, 
                                                                  logits = pred_box_class)
  loss_class = tf.reduce_sum(loss_class * tf.squeeze(obj_mask)) / (N_obj+1e-4)

  return loss_class


def yolo_confidence_loss_square_mask(pred_confidence, pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    true_boxes, obj_mask,
                                    lam_noobj=1, lam_obj=5,
                                    PRECISION=tf.float32):
  # Maximum iou values
  max_ious = yolo_max_iou(pred_xy, pred_wh, true_boxes)
  
  # Get confidence * iou values
  obj_mask_IOU = obj_mask_iou(obj_mask, pred_xy, pred_wh, true_xy, true_wh)
  
  
  
  # get conf_mask
  obj_mask_iou_min = tf.cast(max_ious < 0.6, PRECISION)
  no_obj_mask_iou= (obj_mask_iou_min) * (1-tf.squeeze(obj_mask))
  conf_mask = lam_noobj*no_obj_mask_iou + lam_obj*obj_mask_IOU
  N_obj = tf.reduce_sum(tf.cast(obj_mask>0,dtype=tf.float32))


  confidence_loss = tf.reduce_sum((conf_mask *
                              tf.square(obj_mask_IOU - tf.sigmoid(pred_confidence))) ) / (N_obj + 1e-4) 


  return confidence_loss




def yolo_confidence_loss_bce_mask(pred_confidence, pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    true_boxes, obj_mask,
                                    lam_noobj=1, lam_obj=5,
                                    PRECISION=tf.float32):
  # Maximum iou values
  max_ious = yolo_max_iou(pred_xy, pred_wh, true_boxes)
  obj_mask_iou_min = tf.cast(max_ious < 0.6, PRECISION)

  # Get confidence * iou values
  obj_mask_IOU = obj_mask_iou(obj_mask, pred_xy, pred_wh, true_xy, true_wh)
  
  # get conf_mask
  no_obj_mask_iou= (obj_mask_iou_min) * (1-tf.squeeze(obj_mask))
  conf_mask = lam_noobj*no_obj_mask_iou + lam_obj*obj_mask_IOU
  N_obj = tf.reduce_sum(tf.cast(obj_mask>0,dtype=tf.float32))


  confidence_loss = tf.reduce_sum((conf_mask *
                              tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_confidence, labels=obj_mask_IOU) ) / (N_obj + 1e-4) )

  return confidence_loss



def yolo_confidence_loss_bce_mask2(pred_confidence, pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    obj_mask,
                                    lam_noobj=1, lam_obj=5,
                                    PRECISION=tf.float32):

  # Get confidence * iou values
  obj_mask_IOU = obj_mask_iou(obj_mask, pred_xy, pred_wh, true_xy, true_wh)
  
  # get conf_mask
  conf_mask = lam_noobj*(1-tf.squeeze(obj_mask)) + lam_obj*obj_mask_IOU
  N_obj = tf.reduce_sum(tf.cast(obj_mask>0,dtype=PRECISION))


  confidence_loss = tf.reduce_sum((conf_mask *
                              tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_confidence, labels=obj_mask_IOU) ) / (N_obj + 1e-4) )


  return confidence_loss

# Custom loss function for YOLOv2
class YoloLoss(Loss):

  def __init__(self, anchors,
              num_classes,
              w_coord=1, w_class=1, w_conf=1, lam_obj=5, lam_noobj=1,
              PRECISION = tf.float32, print_loss = False):
    super().__init__()
    self.anchors = anchors
    self.num_classes = num_classes
    self.w_coord = w_coord
    self.w_class = w_class
    self.w_conf = w_conf
    self.lam_obj = lam_obj
    self.lam_noobj = lam_noobj
    self.PRECISION = PRECISION
    self.print_loss = print_loss
  
  @tf.function
  def call(self, y_true, y_pred):
    """
    YOLO localization loss function. No reduction at all, no average

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.

    print_loss : bool, default=False
        If True then use a tf.Print() to print the loss components.

    Returns
    -------
    total_loss : float
        total localization loss across minibatch
    """
    # Inputs
    batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)

    # y_true, true_boxes = y_true
    
    # Predictions
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(
                                                            y_pred, self.anchors)

    # Ground Truths
    true_xy    = y_true[..., 0:2]
    true_wh    = y_true[..., 2:4]
    obj_mask  = y_true[...,4:5]
    true_class = tf.cast(y_true[...,5], tf.int32)

    # Coordinate loss
    coord_loss = yolo_coord_loss(pred_xy, pred_wh, 
                                      true_xy, true_wh,
                                      obj_mask) / batch_size
    
    # Class loss
    class_loss = yolo_class_loss(pred_class_prob, true_class, obj_mask) / batch_size

  
  
    # Condfindence Loss
    # confidence_loss = yolo_confidence_loss_square_mask(pred_confidence, pred_xy, pred_wh, 
    #                                                   true_xy, true_wh,
    #                                                   true_boxes, obj_mask,
    #                                                   lam_noobj=self.lam_noobj, lam_obj=self.lam_obj,
    #                                                   PRECISION=self.PRECISION) / batch_size

    confidence_loss = yolo_confidence_loss_bce_mask2(pred_confidence, pred_xy, pred_wh, 
                                                      true_xy, true_wh,
                                                      obj_mask,
                                                      lam_noobj=self.lam_noobj, lam_obj=self.lam_obj,
                                                      PRECISION=self.PRECISION) / batch_size



    yolo_total_loss = (self.w_coord*coord_loss + self.w_class*class_loss + self.w_conf*confidence_loss )
    
    if self.print_loss:
      tf.print(f"coord_loss={coord_loss}       class_loss={class_loss}     confidence_loss={confidence_loss}")

    return yolo_total_loss