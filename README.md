## CLASSIFIER MODEL
  

### install 

    pip install -r requirements.txt


### inference

     python predict_manta.py --folder_images demo_images --filename_out demo_images_out.csv --filename_mask demo_mask.png

arguments :

- folder_images  : folder with input images
- filename_out   : output csv file with columns filename, manta score, prediction
- filename_model : model , default yolo11cls_manta_640_grayscale3.pt
- filename_mask  : filename of mask (optional if not specified it will analyze full image)
- th             : score threshold, classified as manta if score >= threshold , default=0.97. th between 0 and 1. decrease th to decrese false negative and increase false positives   
  

see the video instructions_for_mask.mp4 to create the mask


### postprocess 

divide images in positive,negative,unknown based on the output csv file from the inference script

     python predict_manta_postprocess.py --folder_images demo_images --filename_csv out.csv --folder_out demo_images

arguments :

- folder_images  : folder with input images
- filename_csv   : csv file with columns filename, manta score, prediction
- folder_out     : output folder with images divided in positive,negative,unknown
- th             : score threshold, classified as manta if score >= threshold

### training with new data




- download the last training dataset from https://drive.google.com/file/d/1urs0XvHI5AchrmBoNC0ngpTSg8ZBtgIE/view?usp=drive_link

- unzip dataset_manta_cls4.zip
  it will have these folders
  folders are

  test

     manta

     non_manta

  train

      manta

      non_manta


- add new images in the correct folders. better to add simila number of images in manta and non manta classes to keep the dataset balanced
     
- zip the new dataset to   dataset_manta_cls4.zip
 
- upload the zip file on your gdrive, get the link of the file and grant access to everyone with the link

- open colab and load the notebook mantatrust_mantaclassification_train.ipynb , go to Runtime/change Runtime type and set GPU 

- in the first cell set the correct gdrive link for the dataset file and model file , the default values are the last model and dataset on my gdrive

- go to Runtime/execute all to run the notebook

- after the training is completed you find the new model in /content/runs/classify/train/weights/best.pt that you can download locally

  see the videos instructions_train1.mp4 instructions_train2.mp4 instructions_train3.mp4









   
