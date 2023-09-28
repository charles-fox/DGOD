import json
import cv2
import numpy as np
from os import walk
import pandas as pd
import random
import os
import glob
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


if __name__ == '__main__':
  
  args = parser_args()
  
  data_json = open(f'ACDC/gt_detection/instancesonly_{args.image_set}_gt_detection.json')
  data_dict = json.load(data_json)

  #Creation of id-->path mapping
  img_path_id_dict = data_dict['images']
  path_data_dict = {}
  for img in img_path_id_dict:
    path_data_dict[img['id']] = img['file_name']

  #Empty bbox list
  bbox_data_dict = {}
  label_data_dict = {}  
  for key in path_data_dict.keys():
    bbox_data_dict[key] = []
    label_data_dict[key] = []

  #Category ID mapping
  category_id_list = data_dict['categories']
  category_data_dict = {}
  for category in category_id_list:
    category_data_dict[category['id']] = category['name']


  annotations_list = data_dict['annotations']
  for annot in annotations_list:
    bbox = annot['bbox']
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
  
    bbox_data_dict[annot['image_id']].append(bbox)
    label_data_dict[annot['image_id']].append(labels_to_ind[category_data_dict[annot['category_id']]])
  
    label_count[category_data_dict[annot['category_id']]] += 1

  df = pd.DataFrame(columns=['image_name', 'BoxesString', 'LabelsString'])  
  for key in bbox_data_dict.keys():
    img_path = path_data_dict[key]
    #img = cv2.imread('/mnt/New/datasets/ACDC/rgb_anon/'+img_path)
    bboxes = bbox_data_dict[key]
    labels = label_data_dict[key]
  
    BoxesString = encode_boxes(bboxes)
    LabelsString = encode_labels(labels)
  
    new_row = {'image_name':img_path, 'BoxesString': BoxesString, 'LabelsString': LabelsString}

    df = pd.concat([df, pd.DataFrame.from_records([new_row])], ignore_index = True)


  print(label_count)
  df = df.reset_index(drop=True)
  df.to_csv('./Annots/acdc_'+args.image_set+'_all.csv')
  print(df.head())

  



