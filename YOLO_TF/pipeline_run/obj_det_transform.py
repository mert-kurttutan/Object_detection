import tensorflow as tf
from pipeline_run import config


@tf.function
def process_image(image):
    img = tf.io.decode_jpeg(image[0], channels=3)
    img = tf.image.resize(img, [config["IMAGE_H"],config["IMAGE_W"]])
    img = tf.cast(img, tf.float32)
    img = img / 255

    return img


@tf.function
def grid_scale_xy(b_xmin, b_ymin, b_width, b_height, width, height):
	'''
	obj:     dictionary containing xmin, xmax, ymin, ymax
	config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
	'''
	center_x = b_xmin + (b_width / 2.)
	center_x = config["GRID_W"] * (center_x / width )
	center_y = b_ymin + (b_height / 2.)
	center_y = config["GRID_H"] * (center_y / height )
	
	return (center_x,center_y)

@tf.function
def grid_scale_wh(b_width, b_height, width, height):
	'''
	obj:     dictionary containing xmin, xmax, ymin, ymax
	config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
	'''    
  
	# unit: grid cell
	center_w = config["GRID_W"]* ( b_width / width)

	# unit: grid cell
	center_h = config["GRID_H"] * (b_height / height)

	return (center_w,center_h)
 


@tf.function
def get_truth(args):  
  (cat_id, b_xmin, b_ymin, 
        b_width, b_height, width, height, box) = args
  width = width[0]
  height = height[0]
  y_batch = tf.zeros([config["GRID_H"], config["GRID_W"], config["BOX"], 4+1+1]) 
  for idx in range(tf.shape(cat_id)[0]):

    center_x, center_y = grid_scale_xy(b_xmin[idx], b_ymin[idx],
                                        b_width[idx], b_height[idx], width, height)


    GRID_X = int(tf.math.floor(center_x))
    GRID_Y = int(tf.math.floor(center_y))

    center_w, center_h = grid_scale_wh(b_width[idx], b_height[idx], width, height)



    best_anchor = tf.cast(box[idx], tf.int32)
                      
    # box: center_x, center_y, center_width, center_height
    indices = [[GRID_Y, GRID_X, best_anchor, 0], [GRID_Y, GRID_X, best_anchor, 1],
               [GRID_Y, GRID_X, best_anchor, 2], [GRID_Y, GRID_X, best_anchor, 3],
               [GRID_Y, GRID_X, best_anchor, 4], [GRID_Y, GRID_X, best_anchor, 5]]

    updates = [center_x, center_y, center_w, center_h, 1., cat_id[idx]]
    y_batch = tf.tensor_scatter_nd_update(y_batch, indices, updates)

  return y_batch



def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs."""
  outputs = {}

  inputs["b_xmin"] = tf.sparse.to_dense(inputs["b_xmin"])
  inputs["b_ymin"] = tf.sparse.to_dense(inputs["b_ymin"])
  inputs["b_width"] = tf.sparse.to_dense(inputs["b_width"])
  inputs["b_height"] = tf.sparse.to_dense(inputs["b_height"])
  inputs["cat_id"] = tf.sparse.to_dense(inputs["cat_id"])
  inputs["box"] = tf.sparse.to_dense(inputs["box"])


  # tf.io.decode_jpeg function cannot be applied on a batch of data.
  # We have to use tf.map_fn
  outputs["x_batch"] = tf.map_fn(
        process_image,
        inputs["image"],
        dtype=tf.float32,
    )

  outputs["y_batch"] = tf.map_fn(
        get_truth,
        (inputs["cat_id"], inputs["b_xmin"], inputs["b_ymin"], 
         inputs["b_width"], inputs["b_height"], inputs["width"], 
         inputs["height"], inputs["box"]),
        dtype=tf.float32)


  return outputs