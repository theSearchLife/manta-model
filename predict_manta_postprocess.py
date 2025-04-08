import os
import pandas as pd
import shutil
import argparse




if __name__=='__main__':
   #parse parameters
   parser = argparse.ArgumentParser()
   
   parser.add_argument("--folder_images", type=str,required=True)
   parser.add_argument("--folder_out", type=str,default='')
   parser.add_argument("--filename_csv", type=str,required=True)
   parser.add_argument("--th", type=float,default=0.5)
   
   args = parser.parse_args()
   
   folder=args.folder_images
   filename_csv=args.filename_csv
   folder_out=args.folder_out
   if len(folder_out)==0:
      folder_out=folder+'_out'
   th=args.th
   folder_out_manta=folder_out+'/'+'positive'
   folder_out_nonmanta=folder_out+'/'+'negative'
   folder_out_unkown=folder_out+'/'+'unknown'
   os.makedirs(folder_out_manta,exist_ok=True)
   os.makedirs(folder_out_nonmanta,exist_ok=True)
   os.makedirs(folder_out_unkown,exist_ok=True)
   df = pd.read_csv(filename_csv)
   for ct in range(len(df)):
      filename= df['filename'].iloc[ct]
      prediction= df['prediction'].iloc[ct]
      score=df['Manta score '].iloc[ct]
      if score>=th:
         prediction='manta'
      else:
         prediction ='non manta'
      #print( filename,prediction)
      if prediction == 'manta':
         path_out=folder_out_manta+'/'+filename
      else:
            if prediction == 'non manta':
               path_out=folder_out_nonmanta+'/'+filename
            else:
               path_out=folder_out_unkown+'/'+filename
      src=folder+'/'+filename
      dst=path_out
      #print(src,dst)
      shutil.copyfile(src,dst)

   print('done', folder_out)