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

  f = open('BDD100K/labels/ins_seg_'+args.image_set+'.json')
  data = json.load(f)

  df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])


  count = 0
  for item in data:
    img_name = item['name']
  
    bboxes = []
    labels = []
    for obj in item['labels']:
  
      if(obj['category'] in labels_to_ind.keys()):
        polygon = np.array(obj['poly2d'][0]['vertices'])
      
        if(polygon.size != 0):
          xmin = int(np.min(polygon[:, 0]))
          xmax = int(np.max(polygon[:, 0]))
          ymin = int(np.min(polygon[:, 1]))
          ymax = int(np.max(polygon[:, 1]))
      
          bboxes.append([xmin, ymin, xmax, ymax])
          labels.append(labels_to_ind[obj['category']])
      
          label_count[obj['category']] += 1
            
    
    BoxesString = encode_boxes(bboxes)
    LabelsString = encode_labels(labels)
  
    new_row = {'image_name':img_name, 'BoxesString': BoxesString, 'LabelsString': LabelsString}
    df = pd.concat([df, pd.DataFrame.from_records([new_row])], ignore_index = True)
 
  

  df = df.reset_index(drop=True)
  df.to_csv('./Annots/bdd10k_'+args.image_set+'_all.csv')
  f.close()

  print(df.head())
  print(label_count)






