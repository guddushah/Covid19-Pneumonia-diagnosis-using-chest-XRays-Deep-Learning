# COVID19 AND PNEUMONIA DIAGNOSIS USING CHEST X-RAYS

## Problem Statement:
#### This project is developed to diagnose the chest X-Rays images data and build a Deep Learning Model to detect COVID19 or PNEUMONIA in patients and ensuring medical person to identify the disease using an android app and perform treatment as early as possible.
#### Dataset: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-imageses

## Project Architecture
- #### tf model built using CNN architecture
- #### Website built in ReactJS to send X-rays images as http request to backend
- #### Backend built using FastAPI framework(mostly used for numpy convertion of the image data)
- #### tf-serving server created using tensorflow/serving docker image (deployed the project file to this image) for serving the http request from website. This server is basically created to control the version management of the model.
- #### Created GCP bucket in google cloud for uploading the model to the cloud and using google sdk for uploading the function in the cloud to serve the requests coming from android app
- #### Created an android app to diagnoising/predicting the class/disease of the uploaded X-Ray image file.
  

