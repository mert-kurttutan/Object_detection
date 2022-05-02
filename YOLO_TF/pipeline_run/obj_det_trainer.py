from typing import Dict, List, Text

from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow_transform as tft

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_transform import TFTransformOutput


from pipeline_run import config, anchors_tf



from src import (obj_det_model)

# Imported files such as taxi_constants are normally cached, so changes are
# not honored after the first import.  Normally this is good for efficiency, but
# during development when we may be iterating code it can be a problem. To
# avoid this problem during development, reload the file.



_LABEL_KEY_list = ["b_xmin", "b_ymin", "b_width", "b_height",
                        "cat_id", "box"]
_LABEL_KEY = "y_batch"

_BATCH_SIZE = 16
_LR = .0003


def load_weight(model, weight_reader):

  weight_reader.reset()
  nb_conv = 23

  for i in range(1, nb_conv):
    conv_layer = model.get_layer('conv_' + str(i))
      
    if i < nb_conv:
      norm_layer = model.get_layer('norm_' + str(i))

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
    layer   = model.layers[-(i+2)] # the last convolutional layer
    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape)/(config["GRID_W"]*config["GRID_H"])
    new_bias   = np.random.normal(size=weights[1].shape)/(config["GRID_H"]*config["GRID_W"])

    layer.set_weights([new_kernel, new_bias])



  return model



def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 20) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      tf_transform_output.transformed_metadata.schema)

def _get_tf_examples_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_inference = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    for _key in _LABEL_KEY_list:
      raw_feature_spec.pop(_key)
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_inference(raw_features)
    logging.info('serve_transformed_features = %s', transformed_features)

    outputs = model(transformed_features)
    # TODO(b/154085620): Convert the predicted labels from the model using a
    # reverse-lookup (opposite of transform.py).
    return {'outputs': outputs}

  return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
    transformed_features = model.tft_layer_eval(raw_features)
    logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn


def export_serving_model(tf_transform_output, model, output_dir):
  """Exports a keras model for serving.
  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    model: A keras model to export for serving.
    output_dir: A directory where the model will be exported to.
  """
  # The layer has to be saved to the model for keras tracking purpases.
  model.tft_layer = tf_transform_output.transform_features_layer()

  signatures = {
      'serving_default':
          _get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
  }

  model.save(output_dir, save_format='tf', signatures=signatures)


def _build_keras_model(tf_transform_output: TFTransformOutput
                       ) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying taxi data.

  Args:
    tf_transform_output: [TFTransformOutput], the outputs from Transform

  Returns:
    A keras Model.
  """
  feature_spec = tf_transform_output.transformed_feature_spec().copy()
  feature_spec.pop(_LABEL_KEY)

  inputs = {}
  for key, spec in feature_spec.items():
    if isinstance(spec, tf.io.VarLenFeature):
      inputs[key] = tf.keras.layers.Input(
          shape=[None], name=key, dtype=spec.dtype, sparse=True)
    elif isinstance(spec, tf.io.FixedLenFeature):
      # TODO(b/208879020): Move into schema such that spec.shape is [1] and not
      # [] for scalars.
      inputs[key] = tf.keras.layers.Input(
          shape=spec.shape or [1], name=key, dtype=spec.dtype)
    else:
      raise ValueError('Spec type is not supported: ', key, spec)
  
  path_to_weight = "./yolov2.weights"
  weight_reader = obj_det_model.WeightReaderDarkNet19(path_to_weight)
  myVGG = obj_det_model.VGGYOLO(inputs=inputs, CLASS=len(config['CLASS']), BOX=config["BOX"])

  myVGG = load_weight(myVGG, weight_reader)
  return myVGG


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, 
                            tf_transform_output, _BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, 
                           tf_transform_output, _BATCH_SIZE)

  model = _build_keras_model(tf_transform_output)

  loss = obj_det_model.YoloLoss(anchors_tf, len(config['CLASS']),
              w_coord=.5, w_class=1, w_conf=.5, lam_obj=5, lam_noobj=1,
              PRECISION = tf.float32, print_loss = False) 
  model.compile(
      loss=loss,
      optimizer=tf.keras.optimizers.Adam(learning_rate=_LR))

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  # Export the model.
  export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)