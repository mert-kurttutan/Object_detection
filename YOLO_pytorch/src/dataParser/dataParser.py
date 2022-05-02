import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
import pylab
import json
import copy

# Image pre-processing
import cv2

# XML Parser
import xml.etree.ElementTree as ET

# COCO json API
from pycocotools.coco import COCO



def box_transform(x_min, y_min, x_max, y_max):
	'''
	Transforms bounding box coordinates in the following way
	[x_min, y_min, x_max, y_max] --> [x_min, y_min, width, height]
	'''

	width = x_max-x_min
	height = y_max-y_min 
	return [x_min, y_min, width, height]



def cat_name_COCO_to_VOC(name):
	'''Changes the label name from VOC to COCO-format'''

	COCO_VOC_correction = {"aeroplane": "airplane", "diningtable": "dining table", 
													"motorbike": "motorcycle", "pottedplant": "potted plant",
													"sofa": "couch", "tvmonitor": "tv"}
	return COCO_VOC_correction.get(name, name)



def parse_page_VOC(ann, ann_dir, image_dir, labels=[]):

	'''
	Parses one page of XML file of Pascal VOC dataset


	Parameters
	----------
	ann : str
			(Relative) Name of the annotation file to be parsed
	ann_dir : str
			Name of annonation directory that contains xml files for annotations of images
	image_dir :  str
			Name of image directory that contains image files 
	labels : list
			List of labels to be considered when forming data
	Returns
	-------
	image_VOC: dict
			Python dictionary which stores of info about each image
	'''

	# Check if it is XML file
	if "xml" not in ann:
			return None
	
	# Represents each individual image of dataset
	image = {'objects':[]}

	tree = ET.parse(ann_dir + ann)
	
	# Iterate through XML file
	for elem in tree.iter():
			# Attache file name to img object
			if 'filename' in elem.tag:
					path_to_image = os.path.join(image_dir, elem.text)
					image['file'] = path_to_image
	
					if not os.path.exists(path_to_image):
							raise ValueError("file does not exist!\n{path_to_image}")
			
			# Attach image width and height
			if 'width' in elem.tag:
					image['width'] = int(elem.text)
			if 'height' in elem.tag:
					image['height'] = int(elem.text)

			# Object Information in XML
			if ("object" in elem.tag) or ("part" in elem.tag):
					# Initiliaze dict to save detected object in the image
					obj = {}

					# Iterate over its children nodes: whole object and its parts
					for child in elem:
							# Name 
							if "name" in child.tag:
									obj['name'] = cat_name_COCO_to_VOC(child.text)

									# Check if object is in provided labels
									if (len(labels) > 0 and obj['name'] not in labels):
											break
									else:
											image['objects'].append(obj)

							# Bounding Box
							# Modifying the obj will modify obj inside the img['object'] list
							if "bndbox" in child.tag:
									for dim in list(child):
											# Iterate through dims of box
											dim_names = ['xmin', 'ymin', 'xmax', 'ymax']
											for dim_nm in dim_names:
													if dim_nm in dim.tag:
															obj[dim_nm] = int(round(float(dim.text)))


									# New coordinate format
									obj['bbox'] = box_transform(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'])

									# Delete old coordinates
									obj.pop('xmin', None); obj.pop('ymin', None); obj.pop('xmax', None); obj.pop('ymax', None)

	return image



def parse_annotation_VOC(ann_dir, image_dir, labels=[]):
	"""
	
	Parses all the available images from given annonation and image folder
	written with VOC annotation

	Parameters
	----------
	ann_dir : str
			Name of annonation directory that contains xml files for annotations of images
	image_dir :  str
			Name of image directory that contains image files 
	num_classes : list
			List of labels to be considered when forming data
	Returns
	-------
	image_VOC: list
			List of python dictionaries which stores of info about each image
	"""

	images_VOC = []

	for ann in sorted(os.listdir(ann_dir)):

		# Parse one page
		image = parse_page_VOC(ann, ann_dir, image_dir, labels)

		# Pass annotation if it has object
		if len(image['objects']) > 0 and (image is not None):
				images_VOC.append(image)
    

	return images_VOC



def COCO_json_img(json_dict):
  '''Extracts the desired image dictionary that maps image id to image format, created from COCO json formate'''
  id_to_image = {}

  for img in json_dict['images']:
    img_obj = {}

    # File name
    img_obj['file'] = img['coco_url']

    # Dims
    img_obj['width'] = img['width']
    img_obj['height'] = img['height']

    # Objects
    img_obj['objects'] = []
    id_to_image[img['id']] = img_obj

  return id_to_image



def COCO_json_cat(json_dict):
  '''Extracts the dictionary that maps category ids to category names, COCO json'''

  id_to_cat = {}
  for elem in json_dict['categories']:
    id_to_cat[elem['id']] = elem['name']  

  return id_to_cat



def attach_objs(id_to_image, id_to_cat, my_json):

  '''
  Attaches the objects kept in  my_json to the images inside id_to_image dictionary
  For object detection/localizaiton, an object consists of:
                                                      1) name
                                                      2) coordinates of bounding box: (x_min, y_min, width, height)
  '''

  for ann in my_json['annotations']:
    img_obj = {}

    img = id_to_image[ann['image_id']]
    img_obj['name'] = id_to_cat[ann['category_id']]
    img_obj['bbox'] = ann['bbox']
    
    img['objects'].append(img_obj )
  
  return id_to_image




def bdd_to_format(json_file, img_folder, is_attributes=False):

	'''
	Returns the array of dictionaries that represent image for object detection problem
	'''

	img_arr = []

	# Iterate through images
	for img in json_file:

		img_obj = {}

		# Default image size for BDD100K dataset
		BDD_HEIGHT = 700
		BDD_WIDTH = 1280

		# Required keys in our format
		img_obj['file'] = img_folder + img['name']
		img_obj['height'] = BDD_HEIGHT
		img_obj['width'] = BDD_WIDTH
		img_obj['objects'] = []

		if is_attributes:
			img_obj['attributes'] = img['attributes']

		obj={}
		# If there is no object json file does not have 'labels' key
		# To prevent key error, use get method
		for obj_bdd in img.get('labels', []):
			obj = {}
			bdd_box = obj_bdd['box2d']

			# Desired format of box coordinates
			obj['bbox'] = box_transform(bdd_box['x1'],bdd_box['y1'],bdd_box['x2'],bdd_box['y2'])
			obj['name'] = obj_bdd['category']

			img_obj['objects'].append(obj)
			if is_attributes:
				obj['attributes'] = obj_bdd['attributes']

		img_arr.append(img_obj)

	return img_arr







