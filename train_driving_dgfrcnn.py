#!/usr/bin/python -tt
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Common imports
import math
import sys
import time
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import argparser
import os

# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision

# Albumentations is used for the Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pytorch import
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import Subset, WeightedRandomSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
#from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
import cv2


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
seed_everything(42)

class DrivingDataset(Dataset):
    #Dataset class applicable for BDD100K, Cityscapes and Foggycityscapes
    def __init__(self, csv_file, root, domain, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied on a sample.
        """
        self.csv_file = csv_file
        
        annotations = pd.read_csv(self.csv_file)

        self.image_path = annotations["image_name"]
        self.root_dir = root
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.labels = [self.decodeLabString(item) for item in annotations["LabelsString"]]
        self.domain = domain
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        imgp = self.root_dir + self.image_path[idx]
        labels = self.labels[idx] 
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Opencv open images in BGR mode by default
        
        # A few boxes in BDD100K are having incorrect annotations. The following two lines will 
        # aid in needed corrections. 
        
        if(len(bboxes) > 0):
          bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img.shape[1]-1)
          bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img.shape[0]-1)
          bboxes[:, 2] = np.clip(bboxes[:, 2], 1, img.shape[1]-1)
          bboxes[:, 3] = np.clip(bboxes[:, 3], 1, img.shape[0]-1)
        
          bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] = bboxes[:, 0][bboxes[:, 0] == bboxes[:, 2]] - 1
          bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] = bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] - 1
          #bboxes[:, 2][bboxes[:, 0] == bboxes[:, 2]] = bboxes[:, 2][bboxes[:, 0] == bboxes[:, 2]] + 1
          #bboxes[:, 3][bboxes[:, 1] == bboxes[:, 3]] = bboxes[:, 1][bboxes[:, 1] == bboxes[:, 3]] + 1       
        
        transformed = self.transform(image=image,bboxes=bboxes,class_labels=labels) #Albumentations can transform images and boxes
        image = (transformed["image"]/255.0).float()
        bboxes = np.array(transformed["bboxes"])
        labels = transformed["class_labels"]
        
        if len(bboxes) > 0:
          bboxes = torch.tensor(bboxes)
          labels = torch.tensor(labels)
        else:
          bboxes = torch.zeros((0,4))
          labels = torch.tensor([])        
          
        #if len(bboxes) > 0:
        #bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        #labels = torch.stack([torch.tensor(item) for item in labels])
        #else:
          #bboxes = torch.zeros((0,4))
          #labels = torch.zeros(1)        
          
        return image, bboxes, labels, self.domain

    def decodeLabString(self, LabelsString):
      """
      Small method to decode the BoxesString
      """
      #labels_to_ind = {'person':1, 'rider': 2,'car': 3,'truck': 4, 'bus':5, 'train':6, 'motorcycle':7, 'bicycle':8}
      if LabelsString == "no_label":
          return np.array([])
      else:
          try:
            labels =  np.array([int(label) for label in LabelsString.split(";")])
            return labels
              
          except:
            print(LabelsString)
            print("Submission is not well formatted. empty boxes will be returned")
            return np.array([])
              
    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          #return np.zeros((0,4))
          return np.array([])
      else:
          try:
              boxes =  np.array([np.array([float(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
              return boxes.astype(np.int32).clip(min=0)
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))


train_transform = A.Compose(
        [
        A.Resize(height=600, width=1200, p=1.0),
        A.HorizontalFlip(p=0.5),     
        ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20)
    )

val_transform = A.Compose([
    #A.Resize(height=600, width=1200, p=1.0),
    ToTensorV2(p=1.0),
],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))



class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)
    
class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(_InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _InsCls(nn.Module):
    def __init__(self, num_cls):
        super(_InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x

class _ImageDAFPN(nn.Module):
    def __init__(self,dim,num_domains):
        super(_ImageDAFPN,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(256, 256, 3, stride=(2,4))
        self.Conv2 = nn.Conv2d(256, 256, 3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, 3, stride=4)
        self.Conv4 = nn.Conv2d(256, 256, 3, stride=3)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        

        torch.nn.init.normal_(self.Conv1.weight, std=0.001)
        torch.nn.init.constant_(self.Conv1.bias, 0)
        torch.nn.init.normal_(self.Conv2.weight, std=0.001)
        torch.nn.init.constant_(self.Conv2.bias, 0)
        torch.nn.init.normal_(self.Conv3.weight, std=0.001)
        torch.nn.init.constant_(self.Conv3.bias, 0)
        torch.nn.init.normal_(self.Conv4.weight, std=0.001)
        torch.nn.init.constant_(self.Conv4.bias, 0)
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=F.sigmoid(self.linear2(x))
        
        return x
                     
class _ImageDA(nn.Module):
    def __init__(self,dim,num_domains):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(2048, 1024, 3, stride=(2,4))
        self.Conv2 = nn.Conv2d(1024, 512, 3, stride=2)
        self.Conv3 = nn.Conv2d(512, 256, 3, stride=2)
        #self.Conv4 = nn.Conv2d(256, 128, 3, stride=2)
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        

        torch.nn.init.normal_(self.Conv1.weight, std=0.001)
        torch.nn.init.constant_(self.Conv1.bias, 0)
        torch.nn.init.normal_(self.Conv2.weight, std=0.001)
        torch.nn.init.constant_(self.Conv2.bias, 0)
        torch.nn.init.normal_(self.Conv3.weight, std=0.001)
        torch.nn.init.constant_(self.Conv3.bias, 0)
        #torch.nn.init.normal_(self.Conv4.weight, std=0.001)
        #torch.nn.init.constant_(self.Conv4.bias, 0)
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        #x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=F.sigmoid(self.linear2(x))
        
        return x

class myresnet50backbone(nn.Module):
  def __init__(self, out_channels):
    super().__init__()
    backbone = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    
    
    self.conv1 = backbone.conv1
    self.bn1 = backbone.bn1
    self.relu = backbone.relu
    self.maxpool = backbone.maxpool
    self.layer1 = backbone.layer1
    self.layer2 = backbone.layer2
    self.layer3 = backbone.layer3
    self.layer4 = backbone.layer4
    self.out_channels = out_channels
    
    for p in self.conv1.parameters(): p.required_grad = False
    for p in self.bn1.parameters(): p.required_grad = False
    for p in self.relu.parameters(): p.required_grad = False
    for p in self.maxpool.parameters(): p.required_grad = False
    for p in self.layer1.parameters(): p.required_grad = False
    for p in self.layer4.parameters(): p.required_grad = True
    for p in self.layer3.parameters(): p.requried_grad = True
    for p in self.layer2.parameters(): p.required_grad = True 

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x
   
        
def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets=list()
    cls_labels = list()
    domain = list()
    
    for i, t, m, d in batch:
        images.append(i)
        targets.append(t)
        cls_labels.append(m)
        domain.append(d)
        
    images = torch.stack(images, dim=0)

    return images, targets, cls_labels, torch.tensor(domain)
    

import fasterrcnn
class DGFRCNN(LightningModule):
    def __init__(self,n_classes, batch_size, exp, reg_weights):
        super(DGFRCNN, self).__init__()
        self.n_classes = n_classes
        self.num_domains = len(tr_datasets)
        self.batch_size = batch_size
        self.exp = exp
        self.reg_weights = reg_weights
        
        self.detector = fasterrcnn.fasterrcnn_resnet50_fpn(min_size=600, max_size=1200, num_classes=self.n_classes, pretrained=True, trainable_backbone_layers=3)
        self.ImageDA = _ImageDAFPN(256, self.num_domains)
        self.InsDA = _InstanceDA(self.num_domains)       
        self.InsCls = nn.ModuleList([_InsCls(n_classes) for i in range(self.num_domains)])
        self.InsClsPrime = nn.ModuleList([_InsClsPrime(n_classes) for i in range(self.num_domains)])
        
        self.base_lr = 2e-3 #Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay=0.0005
        
        self.best_val_acc = 0
        self.log('val_acc', self.best_val_acc)
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])        
        
        self.detector.backbone.register_forward_hook(self.store_backbone_out)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)
        
        self.mode = 0
        self.sub_mode = 0
    
      
    def store_ins_features(self, module, input1, output):
      self.box_features = output
      self.box_labels = input1[1]
      
            
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output

    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training  and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      
      optimizer = torch.optim.SGD([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.ImageDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsCls.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsClsPrime.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}
                                      ],) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'val_acc'}
      
      
      return [optimizer], [lr_scheduler]
      
     
    
    def train_dataloader(self):
      num_train_sample_batches = len(tr_dataset)//self.batch_size
      temp_indices = np.array([i for i in range(len(tr_dataset))])
      np.random.shuffle(temp_indices)
      sample_indices = []
      for i in range(num_train_sample_batches):
  
        batch = temp_indices[self.batch_size*i:self.batch_size*(i+1)]
  
        for index in batch:
          sample_indices.append(index)  #This is for mode 0
  
        if(self.exp == 'dg'):
          for index in batch:		   #This is for mode 1
            sample_indices.append(index)
      
      return torch.utils.data.DataLoader(tr_dataset, batch_size=self.batch_size, sampler=sample_indices, shuffle=False, collate_fn=collate_fn, num_workers=16)      

      
    def training_step(self, batch, batch_idx):
      
      imgs = list(image.cuda() for image in batch[0]) 
      

      targets = []
      for boxes, labels in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = labels.long().cuda()
        targets.append(target)

      # fasterrcnn takes both images and targets for training, returns
      #Detection using source images
      
      if self.mode == 0:
       
        detections = self.detector(imgs, targets)
        loss = sum([loss for detection in detections for loss in detection['losses'].values()])
        
        
        if(self.exp == 'dg'):             
          if(self.sub_mode == 0):
            self.mode = 1
            self.sub_mode = 1
          elif(self.sub_mode == 1):
            self.mode = 2
            self.sub_mode = 2
          elif(self.sub_mode == 2):
            self.mode = 3
            self.sub_mode = 3
          elif(self.sub_mode == 3):
            self.mode = 4
            self.sub_mode = 4  
          else:
            self.sub_mode = 0
            self.mode = 0
        
        
      elif(self.mode == 1):
        
        loss_dict = {}
          
        _ = self.detector(imgs, targets)
        ImgDA_scores = self.ImageDA(self.base_feat['0'])
        loss_dict['DA_img_loss'] = self.reg_weights[0]*F.cross_entropy(ImgDA_scores, batch[3].to(device=0))
        IDA_out = self.InsDA(self.box_features)
        rep_factor = int(IDA_out.shape[0]/self.batch_size)
        ins_labels = batch[3].reshape(self.batch_size,1).repeat(1, rep_factor).reshape(IDA_out.shape[0])       
        loss_dict['DA_ins_loss'] = self.reg_weights[1]*F.cross_entropy(IDA_out, ins_labels.to(device=0))
        
        ExpImgDA_scores =ImgDA_scores.repeat(1, rep_factor).reshape(IDA_out.shape[0], self.num_domains)
        
        loss_dict['Cst_loss'] = self.reg_weights[2]*F.mse_loss(IDA_out, ExpImgDA_scores)       
        loss = sum(loss1 for loss1 in loss_dict.values())
                
        
        self.mode = 0
              
      elif(self.mode == 2): #Without recording the gradients for detector, we need to update the weights for classifier weights
        loss_dict = {}
        loss = []

        
        for index in range(self.num_domains):
          for param in self.InsCls[index].parameters(): param.requires_grad = True
         
        for index in range(len(imgs)):
          with torch.no_grad():
            _ = self.detector([imgs[index]], [targets[index]])
          
          cls_scores = self.InsCls[batch[3][index].item()](self.box_features)
          loss.append(F.cross_entropy(cls_scores, self.box_labels[0])) 

        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
      elif(self.mode == 3): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
    
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsClsPrime[batch[3][index].item()](self.box_features)
          loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
  	  
        loss_dict['cls_prime'] = self.reg_weights[3]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
        
      else: #For Mode 4
      
        loss_dict = {}
        loss = []
        consis_loss = []
        
        for index in range(self.num_domains):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          temp = []
          for i in range(self.num_domains):
            if(i != batch[3][index].item()):
              cls_scores = self.InsCls[i](self.box_features)
              temp.append(cls_scores)
              loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
          consis_loss.append(torch.mean(torch.abs(torch.stack(temp, dim=0) - torch.mean(torch.stack(temp, dim=0), dim=0))))

        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss))) # + torch.mean(torch.stack(consis_loss)))
        loss = sum(loss for loss in loss_dict.values())
        
        self.mode = 0
        self.sub_mode = 0
 	 
      return {"loss": loss}#, "log": torch.stack(temp_loss).detach().cpu()}

    def validation_step(self, batch, batch_idx):
      
      img, boxes, labels, domain = batch
      
      preds = self.forward(img)
      
      targets = []
      for boxes, labels in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = labels.long().cuda() 
        targets.append(target)
      
      try:
        self.metric.update(preds, targets)
      except:
        print(targets)
          
    def on_validation_epoch_end(self):
      
      metrics = self.metric.compute()
      
      self.log('val_acc', metrics['map_50'])
      print(metrics['map_per_class'], metrics['map_50'])
      self.metric.reset()


def parser_args():
  parser = argparse.ArgumentParser(description='DGFRCNN Main Experiments')
  parser.add_argument('--exp', dest='exp',
                      help='non_dg or dg',
                      default='non_dg', type=str)
                      
  parser.add_argument('--source_domains', dest='source_domains',
                      help='',
                      default='ABC', type=str)
                      
  parser.add_argument('--target_domains', dest='target_domains',
                      help='',
                      default='I', type=str)
  
  parser.add_argument('--weights_folder', dest='weights_folder',
                      help='',
                      default='ABC2I', type=str)
                      
  parser.add_argument('--weights_file', dest='weights_file',
                      help='',
                      default='single_source_acdc', type=str)

  parser.add_argument('--reg_weights', nargs = 5, metavar=('a', 'b', 'c', 'd', 'e'), 
                       dest='reg_weights', help='Regularisation constats', type=float)
                      
  return parser.parse_args()
  
  
if __name__ == '__main__':

  args = parser_args()
  
  NET_FOLDER = args.weights_folder
  
  weights_file = args.weights_file  #dgfrcnn (1, 0.5, 0.1) v1--> (0.5, 0.5, 0.1) v2---> (0.5, 0.5, 0.5) v3---> (0.5, 0.5, 1) v4---> (0.5, 0.5, 0.5) learning rate 0.1*base_lr

  if os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt'): 
    detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  else:	
    if not os.path.exists(NET_FOLDER):
      os.mkdir(NET_FOLDER, 0o777)


  # Dataloader design based on input arguments
  # Training Dataset  
  tr_datasets = []
  domain_index = -1
  if 'a' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/acdc_train.csv', root='/mnt/New/datasets/ACDC/rgb_anon/', transform=train_transform, domain=domain_index))
  if 'b' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/bdd10k_train_all.csv', root='/mnt/New/datasets/BDD100K/images/10k/train/', transform=train_transform, domain=domain_index))
  if 'c' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/cityscapes1_clear_all_train.csv', root='/mnt/New/datasets/cityscapes_clear/train/', transform=train_transform, domain=domain_index))
  if 'i' in args.source_domains.lower():
    domain_index = domain_index + 1
    tr_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/idd_train.csv', root='/mnt/New/datasets/IDD/leftImg8bit/train/', transform=train_transform, domain=domain_index))
  
  tr_dataset = torch.utils.data.ConcatDataset(tr_datasets) # Combine all the source domains with their respective domain_index for training
    
  # Validation Dataset
  vl_datasets = []
  domain_index = -1
  if 'a' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/acdc_val.csv', root='/mnt/New/datasets/ACDC/rgb_anon/', transform=val_transform, domain=domain_index))
  if 'b' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/bdd10k_val_all.csv', root='/mnt/New/datasets/BDD100K/images/10k/val/', transform=val_transform, domain=domain_index))
  if 'c' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/cityscapes1_clear_all_val.csv', root='/mnt/New/datasets/cityscapes_clear/val/', transform=val_transform, domain=domain_index))
  if 'i' in args.source_domains.lower():
    domain_index = domain_index + 1
    vl_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/idd_val.csv', root='/mnt/New/datasets/IDD/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  
  vl_dataset = torch.utils.data.ConcatDataset(vl_datasets) # Combine all the source domains with their respective domain_index for validation
  
  # Test Dataset
  test_datasets = []
  domain_index = -1
  if 'a' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/acdc_val.csv', root='/mnt/New/datasets/ACDC/rgb_anon/', transform=val_transform, domain=domain_index))
  if 'b' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/bdd10k_val_all.csv', root='/mnt/New/datasets/BDD100K/images/10k/val/', transform=val_transform, domain=domain_index))
  if 'c' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/cityscapes1_clear_all_val.csv', root='/mnt/New/datasets/cityscapes_clear/val/', transform=val_transform, domain=domain_index))
  if 'i' in args.target_domains.lower():
    domain_index = domain_index + 1
    test_datasets.append(DrivingDataset('/mnt/New/datasets/Annots/idd_val.csv', root='/mnt/New/datasets/IDD/leftImg8bit/val/', transform=val_transform, domain=domain_index))
  
  test_dataset = torch.utils.data.ConcatDataset(test_datasets) # Combine all the source domains with their respective domain_index for Testing


  val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  
  # Instantiating the detector
  detector = DGFRCNN(9, 8, args.exp, args.reg_weights) # Num classes + 1 and batch_size

  
  early_stop_callback= EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')
  checkpoint_callback = ModelCheckpoint(monitor='val_acc', dirpath=NET_FOLDER, filename=weights_file, mode='max')
  
  trainer = Trainer(accelerator="gpu", max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], reload_dataloaders_every_n_epochs=1, num_sanity_val_steps=2)
  trainer.fit(detector, val_dataloaders=val_dataloader)
  
  
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  trainer = Trainer(accelerator="gpu", max_epochs=0, num_sanity_val_steps=-1)
  trainer.fit(detector, val_dataloaders=test_dataloader)
    

  
      
