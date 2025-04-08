## CLASSIFIER MODEL
  
tested with python 3.10

### install 

    pip install -r requirements.txt


### inference

     python predict_manta.py --folder_images demo_images --filename_out demo_images_out.csv

arguments :

- folder_images  : folder with input images
- filename_out   : output csv file with columns filename, manta score, prediction
- filename_model : model , default yolo11cls_manta_640_grayscale2.pt
- th             : score threshold, classified as manta if score >= threshold , default=0.2. th between 0 and 1. decrease th to decrese false negative and increase false positives   
  
  
### postprocess 

divide images in positive,negative,unknown based on the output csv file from the inference script

     python predict_manta_postprocess.py --folder_images demo_images --filename_csv out.csv --folder_out demo_images

arguments :

- folder_images  : folder with input images
- filename_csv   : csv file with columns filename, manta score, prediction
- folder_out     : output folder with images divided in positive,negative,unknown

### training

- download the training dataset

     gdown 1n431PCOY8RXZebY-cSnNycvAvCC4DfWK

     unzip dataset_manta_cls3.zip

- init model 

     from ultralytics import YOLO

     model = YOLO("yolo11n-cls.pt") ## this is for first training.  
     #model=  YOLO(path_to_best_model) # this is to resume training 
     
- run training

     results = model.train(data="/content/dataset_manta_cls3", epochs=100, imgsz=640)
     

- it's possible to add new training data just copying them to the correct folders in the base training dataset.

   the training images must be converted to  grayscale  and optionally can be reduced to width size 640

   the dataset folders are

   test

   -- manta

   -- non_manta

   train

   -- manta

   -- non_manta






   