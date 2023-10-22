# COVID19 AND PNEUMONIA DIAGNOSIS USING CHEST X-RAYS

## Problem Statement:
#### This project is developed to diagnose the chest X-Rays images data and build a Deep Learning Model to detect COVID19 or PNEUMONIA in patients and ensuring medical person to identify the disease using an android app and perform treatment as early as possible.
#### Dataset: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-imageses

## Project Architecture
- #### tf model built using CNN architecture
- #### Website built in ReactJS to send X-rays images as http request to backend
- #### Backend built using FastAPI framework(mostly used for numpy convertion of the image data)
- #### tf-serving server created using tensorflow/serving docker image (deployed the project file to this image) for serving the http request from website. This server is basically created to control the version management of the models.
- #### Created GCP bucket in google cloud for uploading the model to the cloud and using google sdk for uploading the function to serve the requests coming from android app
- #### Created an android app in React Native to diagnoising/predicting the class/disease of the uploaded X-Ray image file.

  ### Website and App Screenshot
 ![chestxray_scrnshot2](https://github.com/guddushah/Covid19-Pneumonia-diagnosis-using-chest-XRays-Deep-Learning/assets/40028193/4e1765b8-35c3-43ea-8a6d-f0ab149d90f6)

  <img src="https://github.com/guddushah/Covid19-Pneumonia-diagnosis-using-chest-XRays-Deep-Learning/assets/40028193/e4ab42bc-3902-4006-9d32-c39f977bdbaf" width="350">

  ### Website Demo
https://github.com/guddushah/Covid19-Pneumonia-diagnosis-using-chest-XRays-Deep-Learning/assets/40028193/5f257959-b7d3-4e76-8077-716b9b141f19


https://github.com/guddushah/Covid19-Pneumonia-diagnosis-using-chest-XRays-Deep-Learning/assets/40028193/431de605-cbab-4ead-8183-d1cb68054024
  

