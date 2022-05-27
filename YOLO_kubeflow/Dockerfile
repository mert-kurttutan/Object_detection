# Copyright 2018 The Kubeflow Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y jq
RUN  apt-get update \
  && apt-get install -y wget
RUN  apt-get update \
  && apt-get install -y zip
  
RUN apt-get update && apt-get install -y python3-opencv

RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud
RUN tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
RUN /usr/local/gcloud/google-cloud-sdk/install.sh

ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
 
COPY . /opt/workdir
RUN python3 -m pip install -r /opt/workdir/requirements.txt
#RUN python3 -m pip install google-cloud-storage
#RUN python3 -m pip install pandas
#RUN python3 -m pip install opencv-python
#RUN python3 -m pip install scikit-image
WORKDIR /opt/workdir


