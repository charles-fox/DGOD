import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
import os
import argparse	

labels_to_ind = {'person':1, 'rider': 2,'car': 3,'truck': 4, 'bus':5, 'train':6, 'motorcycle':7, 'bicycle':8}
label_count = {'person':0, 'rider': 0,'car': 0,'truck': 0, 'bus':0, 'train':0, 'motorcycle':0, 'bicycle':0}

def parser_args():
  parser = argparse.ArgumentParser(description='Convert Annots to CSV')
  parser.add_argument('--image_set', dest='image_set',
                      help='train or val',
                      default='train', type=str)
                                           
  args = parser.parse_args()
  return args
  
def encode_boxes(boxes):

  if len(boxes) >0:
    boxes = [" ".join([str(float(i)) for i in item]) for item in boxes]
    BoxesString = ";".join(boxes)
  else:
    BoxesString = "no_box"
  return BoxesString

def encode_labels(labels):

  if len(labels) >0:
    labels = [" ".join([str(item)]) for item in labels]
    LabelsString = ";".join(labels)
  else:
    LabelsString = "no_label"
  return LabelsString




df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelString'])

if __name__ == '__main__':

  args = parser_args()

  main_path = "IDD/leftImg8bit/"+args.image_set
  labelpath = "IDD/gtFine/"+args.image_set

  flnames = os.listdir(main_path)
  df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])
  
  
  for flname in flnames:
    fnames = os.listdir(main_path+'/'+flname)
    for fname in fnames:
    
      fnumber = fname.split('.')[0].split('_')[0]
      data = json.load(open(labelpath+'/'+flname+'/'+fnumber+'_gtFine_polygons.json'))
    
      bboxes = []
      labels = []
    
      #print(flname+'/'+fname)
      for item in data['objects']:
        if(item['label'] in labels_to_ind.keys()):
          label_count[item['label']] += 1
        
          polygon = np.array(item['polygon'])
        
          if(polygon.size != 0):
            xmin = np.min(polygon[:, 0])
            xmax = np.max(polygon[:, 0])
            ymin = np.min(polygon[:, 1])
            ymax = np.max(polygon[:, 1])
            bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            labels.append(labels_to_ind[item['label']])
	
      BoxesString = encode_boxes(bboxes)
      LabelsString = encode_labels(labels)
     
      new_row = {'image_name':flname + '/' + fname, 'BoxesString': BoxesString, 'LabelsString': LabelsString}

      df = pd.concat([df, pd.DataFrame.from_records([new_row])], ignore_index = True)

  df = df.reset_index(drop=True)
  df.to_csv('./Annots/idd_'+args.image_set+'_all.csv')
  print(df.head())


