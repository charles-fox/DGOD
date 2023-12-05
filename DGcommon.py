from __future__ import absolute_import, division, print_function

import math, sys, time, random, os
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchmetrics
import pytorch_lightning



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

class GRLayer(torch.autograd.Function):
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
    
    
class ImageDAFPN(torch.nn.Module):
    def __init__(self,dim,num_domains):
        super(ImageDAFPN,self).__init__()
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
        x=torch.nn.functional.sigmoid(self.linear2(x))
        return x
        
        
class ImageDA(torch.nn.Module):
    def __init__(self,num_domains):		              #CF: FCOS verson used to have an extra dim arg, but not used, so removed
        super(ImageDA,self).__init__()
#        self.dim=dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(2048, 1024, 3, stride=(2,4))
        self.Conv2 = nn.Conv2d(1024, 512, 3, stride=2)
        self.Conv3 = nn.Conv2d(512, 256, 3, stride=2)
        
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
     
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        return x
          
class DGModel(pytorch_lightning.core.module.LightningModule):
    def __init__(self,n_classes, batch_size, exp, reg_weights, tr_dataset, tr_datasets):
        super().__init__()
        self.tr_dataset=tr_dataset
        self.tr_datasets=tr_datasets
        self.n_classes = n_classes
        self.num_domains = len(self.tr_datasets)
        self.batch_size = batch_size
        self.exp = exp
        self.reg_weights = reg_weights
        self.best_val_acc = 0
        self.log('val_acc', self.best_val_acc)
        self.metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])   
        self.mode = 0
        self.sub_mode = 0
        self.momentum = 0.9 
        
        
    def forward(self, imgs,targets=None):		#same as FCOS
      # Torchvision FasterRCNN returns the loss during training  and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
        
   
    def train_dataloader(self):				#all SAME as FCOS
      num_train_sample_batches = len(self.tr_dataset)//self.batch_size
      temp_indices = np.array([i for i in range(len(self.tr_dataset))])
      np.random.shuffle(temp_indices)
      sample_indices = []
      for i in range(num_train_sample_batches):
        batch = temp_indices[self.batch_size*i:self.batch_size*(i+1)]
        for index in batch:
          sample_indices.append(index)  #This is for mode 0
        if(self.exp == 'dg'):
          for index in batch:		   #This is for mode 1
            sample_indices.append(index)
      return torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.batch_size, sampler=sample_indices, shuffle=False, collate_fn=collate_fn, num_workers=4) #CF was 16, use 4 for lower (12Gb) GPU    
   
       
    def validation_step(self, batch, batch_idx):		#all same as FCOS
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
          
    def on_validation_epoch_end(self):				#all same as FCOS
      metrics = self.metric.compute()
      self.log('val_acc', metrics['map_50'])
      print(metrics['map_per_class'], metrics['map_50'])
      self.metric.reset()
        

