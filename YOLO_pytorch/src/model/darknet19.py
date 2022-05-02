import numpy as np
import torch
import torch.nn.functional as F

from operator import ne
from torch import nn

from collections import OrderedDict
from torchinfo import summary

from ..utils import grid_scale_wh, grid_scale_xy, BestAnchorBoxFinder 



def conv_block(chann_in, chann_out, k_size, stride, pad_value, idx='', is_max=False):
  '''
  Returns Composite module of conv_block: Conv2d + BatchNorm2d + Leaky ReLU
  '''


  layer = nn.Sequential()
  # Conv2d
  layer.add_module(name=f'conv_{idx}',
                  module = nn.Conv2d(in_channels=chann_in, out_channels=chann_out, 
                        kernel_size=k_size, padding=pad_value, stride=stride, bias=False))
  # BatchNorm 2d
  layer.add_module(name=f'norm_{idx}', module = nn.BatchNorm2d(chann_out, track_running_stats=True))
  # Leaky Relu
  layer.add_module(name=f"Leaky_ReLU_{idx}", module=nn.LeakyReLU(negative_slope=0.1))
  # To add max pooling or not
  if is_max:
      layer.add_module(name=f"Max_pool_{idx}", module=nn.MaxPool2d(kernel_size=(2,2)))

  return layer



def repeat_conv_block(param_arr, indices):
  '''
  Repeat the conv_block with list of parameters in indices
  '''
  layers = []
  for param, idx in zip(param_arr, indices):
    is_max, chann_in, chann_out, k_size, stride, pad_value = param
    layers.append(conv_block(chann_in, chann_out, k_size, stride, pad_value, idx, is_max))

  return nn.Sequential(*layers)



class VGGYOLO(torch.nn.Module):
  '''
  Torch Module for constructing VGGYOLO Model based on darknet19 architecture
  
  '''


  def __init__(self, CLASS=10, BOX=7, GRID_H=13, GRID_W=13):
    super().__init__()
    # Parametes for classification problem
    self.CLASS = CLASS
    self.BOX = BOX
    self.GRID_H = GRID_H
    self.GRID_W = GRID_W

    # Arrays to provide indices of conv_block, see their names
    indices_1 = np.arange(1, 14, 1, dtype=np.int32)
    indices_2 = np.arange(14, 21, 1, dtype=np.int32)
    indices_skip = np.arange(21, 22, 1, dtype=np.int32)
    indices_3 = np.arange(22, 23, 1, dtype=np.int32)

    # Parameters of conv_blocks
    param_arr_1 = [
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

    param_arr_2 = [
                    (False, 512, 1024, 3, 1, 'same'),    # 14
                    (False, 1024, 512, 1, 1, 'same'),    # 15
                    (False, 512, 1024, 3, 1, 'same'),    # 16
                    (False, 1024, 512, 1, 1, 'same'),    # 17
                    (False, 512, 1024, 3, 1, 'same'),    # 18
                    (False, 1024, 1024, 3, 1, 'same'),    # 19
                    (False, 1024, 1024, 3, 1, 'same'),     # 20
                ]

    param_arr_skip = [(False, 512, 64, 1, 1, 'same')]      # 21 dont forget to add space_depth
    param_arr_3 = [(False, 1280, 1024, 3, 1, 'same')]      # 22

    # Conv blocks (BatchNorm + ReLU activation added in each block)
    self.part1 = repeat_conv_block(param_arr_1, indices_1)
    self.part2 = repeat_conv_block(param_arr_2, indices_2)
    self.skip_layer = repeat_conv_block(param_arr_skip, indices_skip)               
    self.part3 = repeat_conv_block(param_arr_3, indices_3)

    # Last fully connected layer to make prediction
    self.fc1 = nn.Conv2d(in_channels=1024, out_channels=1024,  # 23
                    kernel_size=1, padding='same', stride=1)
    self.fc2 =  nn.Conv2d(in_channels=1024, out_channels=self.BOX * (4 + 1 + self.CLASS),  # 24
                    kernel_size=1, padding='same', stride=1)


  def forward(self, x):

    out = self.part1(x)
    skip_connection = out

    out = F.max_pool2d(out, kernel_size=(2,2))
    out = self.part2(out)

    skip_connection = self.skip_layer(skip_connection)
    skip_connection = F.pixel_unshuffle(skip_connection, downscale_factor=2)
    out = torch.cat([skip_connection, out], dim=1)
    out = self.part3(out)
    out = self.fc1(out)
    out = F.relu(out)
    out = self.fc2(out)
    

    # -1 is for batch dimension
    # in pytorch, have to explicitly indicate this dim
    out = out.view(-1, 4+1+self.CLASS, self.BOX, self.GRID_H, self.GRID_W)

    return out


def get_cell_grid(grid_w, grid_h, batch_size, box_num, device, PRECISION=torch.float): 
    '''
    Helper function to assure that the bounding box x and y are in the grid cell scale
    == output == 
    for any i=0,1..,batch size - 1
    output[i,:,:,5,3] = array([[3., 5.],
                               [3., 5.],
                               [3., 5.]], dtype=float32)
    '''

    cell_x = torch.reshape( torch.tile(torch.arange(0, grid_w), (grid_h,)), (1, 1, 1, grid_h, grid_w)).type(PRECISION).to(device)
    cell_y = torch.permute(cell_x, (0,1,2,4,3))
    cell_grid = torch.tile(torch.cat((cell_x, cell_y), 1), (batch_size, 1, box_num, 1, 1))
    return(cell_grid) 

def vgg_idx_to_part(i):
  '''
  
  Returns the name of module in VGG model, see the architecture of VGGYOLO model constructed above.
  This mapping is based on this architecture.
  '''
  part = ""
  if i >= 1 and i <= 13:
    child_index = i-1
    part = "part1"

  elif i >=14 and i<=20:
    child_index = i-14
    part = "part2"
  elif i == 21:
    child_index = i-21
    part = "skip_layer"
  elif i == 22:
    child_index = i-22
    part = "part3"
  else:
    raise ValueError("The index for this Architecture should be between 0 and 24, exclusive")
  return part, child_index




def extractLayer(model, i, num_conv):

  '''
  Extracts the convolutional and batch normalization layer of VGG model for a given index i.
  Again, this is based on the architecture VGGYOLO.
  '''

  target_conv = torch.zeros(1)
  target_b_norm = torch.zeros(1)
  
  
  if i < num_conv:
    part_name, child_idx = vgg_idx_to_part(i)
    target_child = model.get_submodule(part_name).get_submodule(f"{child_idx}")
    target_conv = target_child.get_submodule(f"conv_{i}")
    target_b_norm = target_child.get_submodule(f"norm_{i}")

  elif i==num_conv:
    target_conv = model.get_submodule("fc2")
  
  return target_conv, target_b_norm


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
    
    

def iou(pred_xy, pred_wh, 
        true_xy, true_wh):

  
  # Tensors must be on the same device
  device = true_xy.device

  true_wh_half = true_wh / 2.
  true_mins    = true_xy - true_wh_half
  true_maxes   = true_xy + true_wh_half
  
  pred_wh_half = pred_wh / 2.
  pred_mins    = pred_xy - pred_wh_half
  pred_maxes   = pred_xy + pred_wh_half    
  
  intersect_mins  = torch.maximum(pred_mins,  true_mins)
  intersect_maxes = torch.minimum(pred_maxes, true_maxes)
  intersect_wh    = torch.maximum(intersect_maxes - intersect_mins, torch.tensor([0.]).to(device))
  intersect_areas = intersect_wh[:,0,:,:,:] * intersect_wh[:,1,:,:,:]
  true_areas = true_wh[:,0,:,:,:] * true_wh[:,1,:,:,:]
  pred_areas = pred_wh[:,0,:,:,:] * pred_wh[:,1,:,:,:]


  union_areas = pred_areas + true_areas - intersect_areas
  iou_scores  = torch.div(intersect_areas, union_areas + 1e-4) 
  return iou_scores



def yolo_max_iou(pred_xy, pred_wh, true_boxes):

  pred_xy_c = torch.unsqueeze(pred_xy, 2)
  pred_wh_c = torch.unsqueeze(pred_wh, 2)

  true_xy = true_boxes[:,0:2]           # (N_batch, 2, TRUE_BOX_BUFFER, 1, 1, 1)
  true_wh = true_boxes[:,2:4]           # (N_batch, 2, TRUE_BOX_BUFFER, 1, 1, 1)

  iou_scores = iou(pred_xy_c, pred_wh_c, true_xy, true_wh)

  max_ious = torch.max(iou_scores, dim=1).values
  
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
    obj_mask_IOU = iou_scores * torch.squeeze(obj_mask)
    return (obj_mask_IOU)
    
    
    
    
def yolo_head(yolo_pred, anchors, grid=13, threshold=6, PRECISION=torch.float):
  '''
  Returns the correct values for object detection problem, 
  given output of YOLOV2 Network

  Note: pred_class_prob is actually logit value for numerical efficiency, stability
  
  '''
  
  box_num = int(anchors.size()[0]/2)
  device = yolo_pred.device
  
  BATCH_SIZE = yolo_pred.shape[0]
  cell_grid = get_cell_grid(grid,grid,BATCH_SIZE,box_num,device,PRECISION)
  # Relative position of center within grid cell
  pred_box_xy = torch.sigmoid(yolo_pred[:,:2,:,:,:]) + cell_grid

  
  
  tns_exp = torch.exp(torch.tensor([threshold/2])).to(device)-threshold
  # Relative width boxes w.r.t grid cell
  box_arr = torch.permute(torch.reshape(anchors,[1,box_num,2,1,1]), [0,2,1,3,4]).to(device)
  # Rescale the prior boxes to predict shape of object
  # pred_box_wh = torch.exp(yolo_pred[:,2:4,:,:,:]) * box_arr
  # pred_box_wh = (F.elu(yolo_pred[:,2:4,:,:,:])+1) * box_arr
 
  mask = yolo_pred[:,2:4,:,:,:] > threshold
  pred_box_wh = (torch.square(yolo_pred[:,2:4,:,:,:] + tns_exp) * mask + torch.exp(yolo_pred[:,2:4,:,:,:]*(~mask))* (~mask)) * box_arr
  
  # Confidence is a probability
  # pred_box_conf = torch.sigmoid(yolo_pred[:,4,:,:,:])
  # Confidence probabilities, actually logit values, to make it numerically stable
  pred_box_conf = yolo_pred[:,4,:,:,:]

  # Class probabilities, actually logit values, to make it numerically stable
  pred_class_prob = yolo_pred[:,5:,:,:,:]

  return pred_box_xy, pred_box_wh, pred_box_conf, pred_class_prob





def yolo_coord_loss(pred_box_xy, pred_box_wh, 
                    true_box_xy, true_box_wh, 
                    obj_mask):
  
  # Normalization so that when summing over 1 is equal to 1
  # Applied to the other terms in the loss function
  N_obj = torch.sum((obj_mask>0))

  yolo_xy_loss = torch.sum(torch.square(true_box_xy - pred_box_xy) * obj_mask) / (N_obj+1e-4)
  yolo_wh_loss = torch.sum(torch.square(true_box_wh - pred_box_wh) * obj_mask) / (N_obj+1e-4)

  return yolo_wh_loss + yolo_xy_loss



def yolo_class_loss(pred_box_class, true_box_class,
                    obj_mask):

  N_obj = torch.sum((obj_mask>0))

  loss_class = F.cross_entropy(input=pred_box_class, target=true_box_class, reduction="none")
  loss_class = torch.sum(loss_class * obj_mask[:,0]) / (N_obj+1e-4)

  return loss_class

 

def yolo_confidence_loss_square_mask(pred_confidence, pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    true_boxes, obj_mask,
                                    lam_noobj=1, lam_obj=5,
                                    PRECISION=torch.float):
  # Maximum iou values
  max_ious = yolo_max_iou(pred_xy, pred_wh, true_boxes)
  
  # Get confidence * iou values
  obj_mask_IOU = obj_mask_iou(obj_mask, pred_xy, pred_wh, true_xy, true_wh)
  
  
  
  # get conf_mask
  obj_mask_iou_min = (max_ious < 0.6).type(PRECISION)
  no_obj_mask_iou= (obj_mask_iou_min) * (1-torch.squeeze(obj_mask))
  conf_mask = lam_noobj*no_obj_mask_iou + lam_obj*obj_mask_IOU
  N_obj = torch.sum((obj_mask>0))


  confidence_loss = torch.sum((conf_mask *
                              torch.square(obj_mask_IOU - pred_confidence)) ) / (N_obj + 1e-4) 


  return confidence_loss




def yolo_confidence_loss_bce_mask(pred_confidence, pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    true_boxes, obj_mask,
                                    lam_noobj=1, lam_obj=5,
                                    PRECISION=torch.float):
  # Maximum iou values
  max_ious = yolo_max_iou(pred_xy, pred_wh, true_boxes)
  obj_mask_iou_min = (max_ious < 0.6).type(PRECISION)
  
  # Get confidence * iou values
  obj_mask_IOU = obj_mask_iou(obj_mask, pred_xy, pred_wh, true_xy, true_wh)
  
  # get conf_mask
  no_obj_mask_iou= (obj_mask_iou_min) * (1-torch.squeeze(obj_mask))
  conf_mask = lam_noobj*no_obj_mask_iou + lam_obj*obj_mask_IOU
  N_obj = torch.sum((obj_mask>0))


  confidence_loss = torch.sum((conf_mask *
                              F.binary_cross_entropy_with_logits(pred_confidence, obj_mask_IOU, reduction="none") ) / (N_obj + 1e-4) )


  return confidence_loss
  
  
  
  
def yolo_confidence_loss_bce_mask2(pred_confidence, pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    obj_mask,
                                    lam_noobj=1, lam_obj=5,
                                    PRECISION=torch.float):
  
  # Get confidence * iou values
  obj_mask_IOU = obj_mask_iou(obj_mask, pred_xy, pred_wh, true_xy, true_wh)
  
  # get conf_mask
  conf_mask = lam_noobj*(1-torch.squeeze(obj_mask)) + lam_obj*obj_mask_IOU
  N_obj = torch.sum((obj_mask>0))


  confidence_loss = torch.sum((conf_mask *
                              F.binary_cross_entropy_with_logits(pred_confidence, obj_mask_IOU, reduction="none") ) / (N_obj + 1e-4) )


  return confidence_loss





def yolo_loss(args,
              anchors,
              num_classes,
              w_coord=1, w_class=1, w_conf=1, lam_obj=5, lam_noobj=1,
              rescore_confidence=False,
              print_loss=False,
              PRECISION = torch.float):
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
  (yolo_pred, yolo_true, true_boxes) = args
  batch_size = yolo_pred.shape[0]
  
  # Predictions
  pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(
                                                          yolo_pred, anchors)

  # Ground Truths
  true_xy = yolo_true[:,0:2,:,:,:]
  true_wh = yolo_true[:,2:4,:,:,:]
  obj_mask = yolo_true[:,4:5,:,:,:]
  true_class = yolo_true[:,5,:,:,:].type(torch.long)

  # Coordinate loss
  coord_loss = yolo_coord_loss(pred_xy, pred_wh, 
                                    true_xy, true_wh,
                                    obj_mask) / batch_size
  
  # Class loss
  class_loss = yolo_class_loss(pred_class_prob, true_class, obj_mask) / batch_size

 
 
  # Condfindence Loss
  # confidence_loss = yolo_confidence_loss_bce_mask(pred_confidence, pred_xy, pred_wh, 
  #                                                    true_xy, true_wh,
   #                                                 true_boxes, obj_mask,
    #                                                lam_noobj=lam_noobj, lam_obj=lam_obj,
     #                                               PRECISION=PRECISION) / batch_size
                                        
                                        
  confidence_loss = yolo_confidence_loss_bce_mask2(pred_confidence, pred_xy, pred_wh, 
                                                    true_xy, true_wh,
                                                    obj_mask,
                                                    lam_noobj=lam_noobj, lam_obj=lam_obj,
                                                    PRECISION=PRECISION) / batch_size

  yolo_total_loss = (w_coord*coord_loss + w_class*class_loss + w_conf*confidence_loss )
  
  if print_loss:
    print(f"coord_loss={coord_loss}       class_loss={class_loss}     confidence_loss={confidence_loss}")

  return yolo_total_loss
  

