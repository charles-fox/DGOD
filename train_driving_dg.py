#!/usr/bin/python -tt

#Example run:
# python3 train_driving_dg.py --exp dg --source_domains AC --target_domains A --weights_folder AC2A --weights_file ac2a_dgfrcnn --reg_weights 0.5 0.5 0.5 0.05 0.0001
#Here, A,B,C refer to the datasets ADCD, DCC100K and Cityscapes.   
#This command trains on datasets A and C and runs on dataset A.

from __future__ import absolute_import, division, print_function

import math, sys, time, random, os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torchvision
import torchmetrics
import albumentations
import albumentations.pytorch
import pytorch_lightning

import DrivingDataset
import DGcommon
import DGFRCNN
import DGFCOS


def parser_args():
  parser = argparse.ArgumentParser(description='Main Experiments')
  parser.add_argument('--exp', dest='exp',
                      help='non_dg or dg',
                      default='non_dg', type=str)
  parser.add_argument('--source_domains', dest='source_domains',
                      help='Source Domains provided as a string',
                      default='ABC', type=str)
  parser.add_argument('--target_domains', dest='target_domains',
                      help='Target domains provided as string',
                      default='I', type=str)
  parser.add_argument('--weights_folder', dest='weights_folder',
                      help='Name of the weights folder',
                      default='ABC2I', type=str)
  parser.add_argument('--weights_file', dest='weights_file',
                      help='Name of the weights file',
                      default='single_source_acdc', type=str)
  parser.add_argument('--reg_weights', nargs = 5, metavar=('a', 'b', 'c', 'd', 'e'), 
                       dest='reg_weights', help='Regularisation constats', type=float)
  return parser.parse_args()

 
def datasetsFromArguments(source_domains, target_domains): 
  # Dataloader design based on input arguments
  # Training Dataset  
  tr_datasets = []
  domain_index = -1
  if 'a' in source_domains:
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset.DrivingDataset('data/Annots/acdc_train_all.csv', root='data/ACDC/rgb_anon/', transform=train_transform, domain=domain_index))
  if 'b' in source_domains:
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset.DrivingDataset('data/Annots/bdd10k_train_all.csv', root='data/BDD100K/images/10k/train/', transform=train_transform, domain=domain_index))
  if 'c' in source_domains:
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset.DrivingDataset('data/Annots/cityscapes_train_all.csv', root='data/Cityscapes/leftImg8bit/train/', transform=train_transform, domain=domain_index))
  if 'i' in source_domains:
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset.DrivingDataset('data/Annots/idd_train_all.csv', root='data/IDD/leftImg8bit/train/', transform=train_transform, domain=domain_index))
  tr_dataset = torch.utils.data.ConcatDataset(tr_datasets) # Combine all the source domains with their respective domain_index for training
    
  # Validation Dataset
  vl_datasets = []
  domain_index = -1
  if 'a' in source_domains:
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset.DrivingDataset('data/Annots/acdc_val_all.csv', root='data/ACDC/rgb_anon/', transform=val_transform, domain=domain_index))
  if 'b' in source_domains:
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset.DrivingDataset('data/Annots/bdd10k_val_all.csv', root='data/BDD100K/images/10k/val/', transform=val_transform, domain=domain_index))
  if 'c' in source_domains:
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset.DrivingDataset('data/Annots/cityscapes_val_all.csv', root='data/Cityscapes/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  if 'i' in source_domains:
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset.DrivingDataset('data/Annots/idd_val_all.csv', root='data/IDD/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  vl_dataset = torch.utils.data.ConcatDataset(vl_datasets) # Combine all the source domains with their respective domain_index for validation
  
  # Test Dataset
  test_datasets = []
  domain_index = -1
  if 'a' in target_domains:
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset.DrivingDataset('data/Annots/acdc_val_all.csv', root='data/ACDC/rgb_anon/', transform=val_transform, domain=domain_index))
  if 'b' in target_domains:
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset.DrivingDataset('data/Annots/bdd10k_val_all.csv', root='data/BDD100K/images/10k/val/', transform=val_transform, domain=domain_index))
  if 'c' in target_domains:
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset.DrivingDataset('data/Annots/cityscapes_val_all.csv', root='data/Cityscapes/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  if 'i' in target_domains:
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset.DrivingDataset('data/Annots/idd_val_all.csv', root='data/IDD/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  test_dataset = torch.utils.data.ConcatDataset(test_datasets) # Combine all the source domains with their respective domain_index for Testing
  
  return (tr_dataset, tr_datasets, vl_dataset, test_dataset)



if __name__ == '__main__':
  SEED=42
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
  pytorch_lightning.seed_everything(SEED)

  args = parser_args()
  NET_FOLDER = args.weights_folder
  weights_file = args.weights_file  

  source_domains = args.source_domains.lower()
  target_domains = args.target_domains.lower()
  
  
  train_transform = albumentations.Compose(
        [
        albumentations.Resize(height=600, width=1200, p=1.0),
        albumentations.HorizontalFlip(p=0.5),     
        albumentations.pytorch.ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=albumentations.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20)
    )

  mymodel="DGRCNN"   #TODO make this a cmd line arg

  if mymodel=="DGRCNN":
      val_transform = albumentations.Compose([
        albumentations.pytorch.ToTensorV2(p=1.0),],p=1.0,bbox_params=albumentations.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))
  elif mymodel=="DGFCOS":
      val_transform = albumentations.Compose([
        albumentations.Resize(height=600, width=1200, p=1.0),
        albumentations.pytorch.ToTensorV2(p=1.0),],p=1.0,bbox_params=albumentations.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))

  (tr_dataset, tr_datasets, vl_dataset, test_dataset) = datasetsFromArguments(source_domains, target_domains)

  val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=DGcommon.collate_fn)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=DGcommon.collate_fn)
  
  detector = DGFRCNN.DGFRCNN(9, 8, args.exp, args.reg_weights,  tr_dataset, tr_datasets)        #**CREATING THE DETECTOR**
  
  if os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt'): 
    detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  else:	
    if not os.path.exists(NET_FOLDER):
      os.mkdir(NET_FOLDER, 0o777)
 
  early_stop_callback= pytorch_lightning.callbacks.early_stopping.EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')
  checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(monitor='val_acc', dirpath=NET_FOLDER, filename=weights_file, mode='max')
  
  trainer = pytorch_lightning.Trainer(accelerator="gpu", max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], reload_dataloaders_every_n_epochs=1, num_sanity_val_steps=2)
  trainer.fit(detector, val_dataloaders=val_dataloader)
  
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  trainer = pytorch_lightning.Trainer(accelerator="gpu", max_epochs=0, num_sanity_val_steps=-1)
  trainer.fit(detector, val_dataloaders=test_dataloader)
