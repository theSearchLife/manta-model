import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from ultralytics import YOLO
import argparse

def preprocess_640_gray(img):
  target_width=640
  scale=target_width/img.shape[1]
  target_height=int(scale*img.shape[0])
  img=cv2.resize(img,(target_width,target_height))
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  return img


def get_mask_binary(mask):
    b,g,r=cv2.split(mask)
    mask_bin=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
    mask_bin[(b>200) & (g>200) & (r>200)]=1
    target_width=640
    scale=target_width/mask.shape[1]
    target_height=int(scale*mask.shape[0])
    mask_bin=cv2.resize(mask_bin,(target_width,target_height), interpolation=cv2.INTER_NEAREST)
    return mask_bin


if __name__=='__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename_model", type=str,default='yolo11cls_manta_640_grayscale3.pt')
    parser.add_argument("--folder_images", type=str,required=True)
    parser.add_argument("--filename_out", type=str,required=True)
    parser.add_argument("--filename_mask", type=str,default='')
    parser.add_argument("--th", type=float,default=0.97)
    
    
    args = parser.parse_args()
    
    filename_model=args.filename_model
    folder=args.folder_images
    filename_out=args.filename_out
    filename_mask=args.filename_mask
    model = YOLO(  filename_model)
    th=args.th

    print('start processing', filename_model,folder,th)
    
    classes=['manta','non manta']
    scores=[]
    filenames_test=[]
    predictions=[]
    files=sorted(os.listdir(folder))
    ct=0
    
    mask_bin=np.zeros((640,640),dtype='uint8')
    if len(filename_mask)>0:
      if os.path.exists(filename_mask):
        print('using mask', filename_mask)
        mask=cv2.imread(filename_mask)
        mask_bin=get_mask_binary(mask)
          

    for f in files:
        if ct%100==0:
          print('processing',f,ct,len(files))
        filename_fullpath=folder+'/'+f
        img=cv2.imread(filename_fullpath)
        img=preprocess_640_gray(img)
        mask_bin=cv2.resize(mask_bin,(img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img[mask_bin>0]=0
        
        results = model(img,verbose=False,augment=False)

        score= round(float(results[0].probs.data[0]),3)
        if score >=th:
           prediction='manta'
        else:
           prediction='non manta'
           
        scores.append(score)
        predictions.append(prediction)
        filenames_test.append(f)
        ct+=1

    dict = {'filename': filenames_test, 'Manta score ': scores, 'prediction': predictions }
    df = pd.DataFrame(dict)
    df.to_csv(filename_out)

    print('saved ', filename_out)