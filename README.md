# YOLOv2 Model for Object Detection

This is a repository of my implementation of YOLOv2 model based on DarkNet19 architecture. Both pytorch and tensorflow version is implemented.

# Description

It contains two main parts:

- YOLO_pytorch:
  - This folder contains the pytorch implementation of darknet19 model for object detection. It also contains hyperparameter tuning using optuna with early stoppin

- YOLO_TF:
  - This folder contains tensorflow implementation of darknet19 model for object detection. It contains hyperparater tunini with kerastuner.
  - It also contains implementatino of deployment pipeline using TFX framework.

- YOLO_kubeflow:
  - It contains Kubeflow pipeline implementation for object detection model
  - Pipeline consists of extraction, transformation, training and deployment
  - TO DO: 
    - Generalize the data preprocessing step to extract other datasets (other than VOC)
    - Add deployment part once kubeflow tf serving documentation is up-to-date.
Source:
- https://www.kubeflow.org/docs/distributions/gke/deploy/
- https://www.tensorflow.org/tfx/tutorials
- https://fairyonice.github.io/tag/object-detection-using-yolov2-on-pascal-voc2012-series.html
