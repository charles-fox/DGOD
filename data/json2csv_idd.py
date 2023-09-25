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
    
      print(flname+'/'+fname)
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
  df.to_csv('./Annots/idd_'+args.image_set+'.csv')
  print(df.head())

"""
f = open('../datasets/bdd100k/labels/det_20/det_'+image_set+'.json')
data = json.load(f)


#df = [pd.DataFrame(columns=['image_name', 'BoxesString']) for index in range(len(weather.keys()))]
df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])

#weather_df = {}
#for key in weather.keys():
  #weather_df[key] = pd.DataFrame(columns=['image_name', 'BoxesString'])

for item in data:
  img_name = item['name']
  
  bboxes = []
  labels = []
  if('labels' in item.keys()):
    for obj in item['labels']:
      if(obj['category'] in labels_to_ind.keys()):
        bboxes.append([int(obj['box2d']['x1']), int(obj['box2d']['y1']), int(obj['box2d']['x2']), int(obj['box2d']['y2'])])
        labels.append(labels_to_ind[obj['category']])
        
  BoxesString = encode_boxes(bboxes)
  LabelsString = encode_labels(labels)
  
  new_row = {'image_name':img_name, 'BoxesString': BoxesString, 'LabelsString': LabelsString}
  #The following line can be uncommented if the weather specific annotations are needed. 
  #weather_df[item['attributes']['weather']] = weather_df[item['attributes']['weather']].append(new_row, ignore_index=True)
  df = pd.concat([df, pd.DataFrame.from_records([new_row])], ignore_index = True)
  #df.append(new_row, ignore_index=True)

df = df.reset_index(drop=True)
df.to_csv('./Annots/bdd100k_'+image_set+'_car.csv')
f.close()
"""
"""  
for key in weather_df.keys():
  weather_df[key] = weather_df[key].reset_index(drop=True)  
  weather_df[key].to_csv('./Annots/bdd100k_'+key+'_val_car.csv')
"""


#df = pd.read_csv('bdd100k_clear_val_car.csv') 
#img_path = '/home/ajeffs/CVPR_2021_codes/datasets/BDD100K/bdd100k_det_20_labels_trainval/bdd100k/images/100k/val/'+df['image_name'][12]



#weather = item['attributes']['weather']
#timeofday = item['attributes']['timeofday']
#scene = item['attributes']['scene']
"""  
  bboxes = []
  if('labels' in item.keys()):
    for obj in item['labels']:
      bboxes.append([obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']])
      if(obj['category']  in categories.keys()):
        categories[obj['category']] = categories[obj['category']] + 1
      else:
        categories[obj['category']] = 1
  BoxesString = encode_boxes(bboxes)
      
print(categories)
print(len(weather.keys()))
"""
"""
for item in data:
  img_name = item['name']
  weather = item['attributes']['weather']
  timeofday = item['attributes']['timeofday']
  scene = item['attributes']['scene']
  
  bboxes = []
  for obj in item['labels']:
    bboxes.append([obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']])
    if(obj['category']  in categories.keys()):
      categories[obj['category']] = categories[obj['category']] + 1
    else:
      categories[obj['category']] = 1
      
  new_row = {'image_name':img_name, 'boxesString': BoxesString, 'weather':weather, 'timeofday':timeofday, 'scene':scene}
  val_df = val_df.append(new_row, ignore_index=True)
  
f.close()
"""  

        
          
     
          

#val_df = val_df.reset_index(drop=True)

#train_df.to_csv('cityscapes_foggy_train_car.csv')  
#val_df.to_csv('cityscapes_foggy_val_car.csv')

#print(len(train_df))
#print(len(val_df))
