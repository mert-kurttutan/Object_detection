import copy
import torch
import cv2
import skimage.io as io
import numpy as np
from ..utils import BestAnchorBoxFinder, grid_scale_wh, grid_scale_xy

# TRANSFORMS

class Rescale(object):
  """Rescale the image in a sample to a given size.

  Args:
  output_size (tuple or int): Desired output size. If tuple, output is
  matched to output_size. If int, smaller of image edges is matched
  to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size, norm=False):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size
    self.norm = norm

  def __call__(self, image, sample):
    # Not to change original annotation
    # Memory overhead is not a problem since these are small objects, temporary
    sample_c = copy.deepcopy(sample)

    # Change the size of image
    h, w = image.shape[:2]

    # Compute new dims
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)


    # h and w are swapped for landmarks because for images,
    # x and y axes are axis 1 and 0 respectively
    image = cv2.resize(image, (new_w, new_h))

    if not self.norm:
        image = image / 255.

    sample_c['height'], sample_c['width'] = new_h, new_w
    # Adjust the size of objects
    if "objects" in sample_c.keys():
      for obj in sample_c['objects']:
        x_min = obj['bbox'][0]*(new_w/w)
        y_min = obj['bbox'][1]*(new_h/h)
        b_width = obj['bbox'][2]*(new_w/w)
        b_height = obj['bbox'][3]*(new_h/h)
        obj['bbox'] = [x_min, y_min, b_width, b_height]

    return image, sample_c

class stochastic_Gaussian_Blur(object):


	def __init__(self):

		self.mye = 3



	def __call__(self, image, sample):


		sample_c = copy.deepcopy(sample)



		return image, sample_c



# DataSet and Data Loader
class YOLOData(torch.utils.data.Dataset):

  def __init__(self, images, detect_info, norm=None, shuffle=True):
    '''
    config : dictionary containing necessary hyper parameters for traning. e.g., 
        {
        'IMAGE_H'         : 416, 
        'IMAGE_W'         : 416,
        'GRID_H'          : 13,  
        'GRID_W'          : 13,
        'LABELS'          : ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
                              'bus',        'car',      'cat',  'chair',     'cow',
                              'diningtable','dog',    'horse',  'motorbike', 'person',
                              'pottedplant','sheep',  'sofa',   'train',   'tvmonitor'],
        'ANCHORS'         : array([ 1.07709888,   1.78171903,  
                                    2.71054693,   5.12469308, 
                                    10.47181473, 10.09646365,  
                                    5.48531347,   8.11011331]),
        'BATCH_SIZE'      : 16,
        'TRUE_BOX_BUFFER' : 50,
        }
    
    '''

    self.detect_info = detect_info
    self.detect_info["box_dim"] = int(len(self.detect_info['anchors'])/2)
    self.detect_info["class_dim"] = len(self.detect_info['classes'])
    self.detect_info["channel_dim"] = 3
    self.images = images
    self.bestAnchorBoxFinder = BestAnchorBoxFinder(detect_info['anchors'])
    # self.imageReader = ImageReader(width=detect_info['image_w'], height=detect_info['image_h'], norm=norm)
    self.shuffle = shuffle

    if self.shuffle:
      np.random.shuffle(self.images)

  # the length of the input data
  def __len__(self):
    return len(self.images)

  
  def __getitem__(self, idx):

    img_instance = self.images[idx]

    # Load the original image, from url or local file
    if "http" in img_instance['file']:
        img_arr = io.imread(img_instance['file'])
    else:
        img_arr = cv2.imread(img_instance['file'])
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    yolo_scale = Rescale((self.detect_info['image_h'], self.detect_info['image_w']))

    img_arr, img_instance_c = yolo_scale(img_arr, img_instance)
    all_objs = img_instance_c['objects']

    # Input data: channel_dim, height_dim, width_dim, compatible with conv2d of pytorch
    x_batch = np.zeros((self.detect_info["channel_dim"], self.detect_info['image_h'], self.detect_info['image_w']))
    b_batch = np.zeros((4, self.detect_info['TRUE_BOX_BUFFER'], 1, 1, 1))

    # Desired network output: (Anchor box idx, object location=4, 1=proba of detection, classes, height dim, width dim)
    y_batch = np.zeros((4+1+1, self.detect_info['box'], self.detect_info['grid_h'], self.detect_info['grid_w'], ))   
    true_box_index = 0

    for obj in all_objs:
      if obj['name'] in self.detect_info['classes']:
        center_x, center_y = grid_scale_xy(obj, self.detect_info)

        GRID_X = int(np.floor(center_x))
        GRID_Y = int(np.floor(center_y))

        if GRID_X < self.detect_info['grid_w'] and GRID_Y < self.detect_info['grid_h']:
          obj_indx  = self.detect_info['classes'].index(obj['name'])
          center_w, center_h = grid_scale_wh(obj,self.detect_info)
          box = [center_x, center_y, center_w, center_h]
          best_anchor, _ = self.bestAnchorBoxFinder(center_w, center_h)
                  
          # box: center_x, center_y, center_width, center_height
          y_batch[0:4, best_anchor, GRID_Y, GRID_X] = box
          # Confidence is 1
          y_batch[4, best_anchor, GRID_Y, GRID_X] = 1.
          # Class probability = 1
          y_batch[5, best_anchor, GRID_Y, GRID_X] = int(obj_indx)
          
          # assign the true box to b_batch
          b_batch[:, true_box_index, 0, 0, 0] = box
          
          true_box_index += 1
          true_box_index = true_box_index % self.detect_info['TRUE_BOX_BUFFER']

    # index in pytorch conv2d is (Channel, Height, Width)
    x_batch = img_arr.transpose((2,0,1))

    return [x_batch, b_batch], y_batch

