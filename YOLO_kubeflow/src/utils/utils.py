import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle





## Object Detection Visualization

def get_grids_ticks(img_array, grid_dims):
  '''
  Returns the grid lines and labels to be used for visualization, 
  see: https://www.delftstack.com/howto/matplotlib/set-matplotlib-grid-interval/
  '''

  # Number of grids in each dims
  grid_x, grid_y = grid_dims

  # Label based on grid index
  idx_to_label = lambda idx: f"grid_{idx}"
  idx_to_label = np.vectorize(idx_to_label)

  # Grid lines to appear on the plot
  grid_lines_x = np.linspace(0, img_array.shape[1], grid_x+1)
  grid_lines_y = np.linspace(0, img_array.shape[0], grid_y+1)


  # Labels of grid cells
  x_labels = idx_to_label(np.arange(grid_x))
  y_labels = idx_to_label(np.arange(grid_y))


  # Compute coordinate of ticks for grid labels
  d_x = grid_lines_x[1] - grid_lines_x[0]
  d_y = grid_lines_y[1] - grid_lines_y[0]
  grid_ticks_x = grid_lines_x[:-1] + d_x/2
  grid_ticks_y = grid_lines_y[:-1] + d_y/2


  return grid_lines_x, grid_lines_y, grid_ticks_x, grid_ticks_y, x_labels, y_labels





def fig_with_grid(img_array, grid_dims):
  '''
  Returns plot of image, img_array, with grids placed on it, described by grid_dims
  Used for grid_based object detection/localizaiton, e.g. YOLO algorithm
  '''

  fig, ax = plt.subplots(figsize=(15,15))
  grid_lines_x, grid_lines_y, grid_ticks_x, grid_ticks_y, x_labels, y_labels = get_grids_ticks(img_array, grid_dims)

  # Image
  ax.imshow(img_array)

  # Grid lines
  ax.set_xticks(grid_ticks_x)
  ax.set_yticks(grid_ticks_y)

  # Grid labels
  ax.set_xticks(grid_lines_x, minor=True)
  ax.set_yticks(grid_lines_y, minor=True)
  ax.set_xticklabels(x_labels)
  ax.set_yticklabels(y_labels)
  ax.grid(True, which="minor")

  return fig, ax


def draw_bbox(ax, sample, ec='g', lw=5):

  for obj in sample['objects']:
    bbox = obj['bbox']
    ax.add_patch( Rectangle(bbox[:2],
                          bbox[2], bbox [3],
                        fc ='none', 
                        ec =ec,
                        lw = lw) )



class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, confidence=None,classes=None):
		self.xmin, self.ymin = xmin, ymin
		self.xmax, self.ymax = xmax, ymax
		## the code below are used during inference
		# probability
		self.confidence = confidence
		# class probaiblities [c1, c2, .. cNclass]
		self.set_class(classes)
			
	def set_class(self,classes):
		self.classes = classes
		self.label_idx = np.argmax(self.classes) 
			
	def get_label(self):  
		return self.label_idx
	
	def get_score(self, is_conf=False):
		if is_conf:
			return self.classes[self.label_idx]*self.confidence
		else:
			return self.classes[self.label_idx]






def interval_intersect(interval_a, interval_b):
	'''
	Calculates the intersection between two intervals (line segments)
	'''
	a1, a2 = interval_a
	b1, b2 = interval_b

	return max(0, min(a2,b2) - max(a1,b1))


def calculate_iou(box1: BoundBox, box2: BoundBox):
	'''
	Calculates intersection over union of 2 boxes
	'''
	intersect_w = interval_intersect([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = interval_intersect([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  

	intersect = intersect_w * intersect_h

	b1_area = (box1.ymax-box1.ymin) * (box1.xmax-box1.xmin)
	b2_area = (box2.ymax-box2.ymin) * (box2.xmax-box2.xmin)

	union = b1_area + b2_area - intersect

	return intersect / union



class BestAnchorBoxFinder(object):

	def __init__(self, ANCHORS):
		'''
		ANCHORS: a np.array of even number length e.g.

		_ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
								2,4, ##  width=2, height=4,  tall large anchor box
								1,1] ##  width=1, height=1,  small anchor box
		'''
		self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
										for i in range(int(len(ANCHORS)//2))]
			

    

	def __call__(self, width, height):
		'''
		Finds the best anchor box for a given dim=(height, widht)
		'''

		best_anchor = -1
		max_iou     = -1
		shifted_box = BoundBox(0, 0, width, height)

		# Iterate through anchor boxes to find best anchor, i.e. highest iou
		for i, anchor in enumerate(self.anchors):
			iou = calculate_iou(shifted_box, anchor)
			if iou > max_iou:
				best_anchor = i
				max_iou = iou
		return (best_anchor,max_iou)  
    

def grid_scale_xy(obj, config):
	'''
	obj:     dictionary containing xmin, xmax, ymin, ymax
	config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
	'''
	bbox = obj['bbox']
	center_x = bbox[0] + (bbox[2] / 2.)
	center_x = config['grid_w'] * (center_x / float(config['image_w']) )
	center_y = bbox[1] + (bbox[3] / 2.)
	center_y = config['grid_h'] * (center_y / float(config['image_h']) )
	
	return (center_x,center_y)

def grid_scale_wh(obj, config):
	'''
	obj:     dictionary containing xmin, xmax, ymin, ymax
	config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
	'''    
	bbox = obj['bbox']
	# unit: grid cell
	center_w = config['grid_w'] * ( bbox[2] / float(config['image_w'] ))

	# unit: grid cell
	center_h = config['grid_h'] * ( bbox[3] / float(config['image_h'] ))

	return(center_w,center_h)