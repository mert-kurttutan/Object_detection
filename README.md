# YOLOv2 Model for Object Detection

This is a repository of my implementation of YOLOv2 model based on DarkNet19 architecture. Both pytorch and tensorflow version is implemented.
During preparation of this project, I greatly benefitted from <a href=https://fairyonice.github.io/Part_1_Object_Detection_with_Yolo_for_VOC_2014_data_anchor_box_clustering.html> this online post </a> implemented in tensorflow v1.

# Description

It contains two main parts:

- YOLO_pytorch:
  - This folder contains the pytorch implementation of darknet19 model for object detection. It also contains hyperparameter tuning using optuna with early stoppin

- YOLO_TF:
  - This folder contains tensorflow implementation of darknet19 model for object detection. It contains hyperparater tunini with kerastuner.
  - It also contains implementatino of deployment pipeline using TFX framework.
