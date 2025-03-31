## CLASSIFIER MODEL
  
tested with python 3.10

### install 

    pip install -r requirements.txt


### inference

     python predict_manta.py --folder_images demo_images --filename_out demo_images_out.csv

arguments :

- folder_images  : folder with input images
- filename_out   : output csv file with columns filename, manta score, prediction
- filename_model : model , default yolo11cls_manta_640_grayscale.pt
- th             : score threshold, classified as manta if score >= threshold , default=0.2
  
  
### postprocess 

divide images in positive,negative,unknown based on the output csv file from the inference script

     python predict_manta_postprocess.py --folder_images demo_images --filename_csv out.csv --folder_out demo_images

arguments :

- folder_images  : folder with input images
- filename_csv   : csv file with columns filename, manta score, prediction
- folder_out     : output folder with images divided in positive,negative,unknown






   