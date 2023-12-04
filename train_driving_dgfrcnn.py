#!/usr/bin/python -tt

#Example run:
# python3 train_driving_dgfrcnn.py --exp dg --source_domains AC --target_domains A --weights_folder AC2A --weights_file ac2a_dgfrcnn --reg_weights 0.5 0.5 0.5 0.05 0.0001
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

import fasterrcnn
import DrivingDataset


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
    
class _InstanceDA(torch.nn.Module):
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

class _InsClsPrime(torch.nn.Module):
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

class _InsCls(torch.nn.Module):
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

class _ImageDAFPN(torch.nn.Module):
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
        x=torch.nn.functional.sigmoid(self.linear2(x))
        return x
                     
class _ImageDA(torch.nn.Module):
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



class DGFRCNN(pytorch_lightning.core.module.LightningModule):
    def __init__(self,n_classes, batch_size, exp, reg_weights, tr_dataset, tr_datasets):
        super(DGFRCNN, self).__init__()
        self.tr_dataset=tr_dataset
        self.tr_datasets=tr_datasets

        self.n_classes = n_classes
        self.num_domains = len(self.tr_datasets)
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
        self.metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])        
        
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

    def training_step(self, batch, batch_idx):
      imgs = list(image.cuda() for image in batch[0]) 
      targets = []
      for boxes, labels in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = labels.long().cuda()
        targets.append(target)

      #the four modes below correspond to the stages decriped in AAAI paper just after eqn 10
      #"we initially freeze ... classified accurately by all "

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
        loss_dict['DA_img_loss'] = self.reg_weights[0]*torch.nn.functional.cross_entropy(ImgDA_scores, batch[3].to(device=0))
        IDA_out = self.InsDA(self.box_features)
        rep_factor = int(IDA_out.shape[0]/self.batch_size)
        ins_labels = batch[3].reshape(self.batch_size,1).repeat(1, rep_factor).reshape(IDA_out.shape[0])       
        loss_dict['DA_ins_loss'] = self.reg_weights[1]*torch.nn.functional.cross_entropy(IDA_out, ins_labels.to(device=0))
        ExpImgDA_scores =ImgDA_scores.repeat(1, rep_factor).reshape(IDA_out.shape[0], self.num_domains)
        loss_dict['Cst_loss'] = self.reg_weights[2]*torch.nn.functional.mse_loss(IDA_out, ExpImgDA_scores)       
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
          loss.append(torch.nn.functional.cross_entropy(cls_scores, self.box_labels[0])) 
        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())
        self.mode = 0

      elif(self.mode == 3): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsClsPrime[batch[3][index].item()](self.box_features)
          loss.append(torch.nn.functional.cross_entropy(cls_scores, self.box_labels[0]))
        loss_dict['cls_prime'] = self.reg_weights[3]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())
        self.mode = 0
        
      else: #For Mode 4
        loss_dict = {}
        loss = []
        for index in range(self.num_domains):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          for i in range(self.num_domains):
            if(i != batch[3][index].item()):
              cls_scores = self.InsCls[i](self.box_features)
              loss.append(torch.nn.functional.cross_entropy(cls_scores, self.box_labels[0]))
        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss))) 
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





######################




def parser_args():
  parser = argparse.ArgumentParser(description='DGFRCNN Main Experiments')
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
    tr_datasets.append(DrivingDataset('data/Annots/idd_train_all.csv', root='data/IDD/leftImg8bit/train/', transform=train_transform, domain=domain_index))
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
  if os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt'): 
    detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  else:	
    if not os.path.exists(NET_FOLDER):
      os.mkdir(NET_FOLDER, 0o777)
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

  val_transform = albumentations.Compose([
    #albumentations.Resize(height=600, width=1200, p=1.0),
    albumentations.pytorch.ToTensorV2(p=1.0),],p=1.0,bbox_params=albumentations.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))


  (tr_dataset, tr_datasets, vl_dataset, test_dataset) = datasetsFromArguments(source_domains, target_domains)



  val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn)
  
  detector = DGFRCNN(9, 8, args.exp, args.reg_weights,  tr_dataset, tr_datasets)        #**CREATING THE DETECTOR**
  
  early_stop_callback= pytorch_lightning.callbacks.early_stopping.EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')
  checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(monitor='val_acc', dirpath=NET_FOLDER, filename=weights_file, mode='max')
  
  trainer = pytorch_lightning.Trainer(accelerator="gpu", max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], reload_dataloaders_every_n_epochs=1, num_sanity_val_steps=2)
  trainer.fit(detector, val_dataloaders=val_dataloader)
  
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
  trainer = Trainer(accelerator="gpu", max_epochs=0, num_sanity_val_steps=-1)
  trainer.fit(detector, val_dataloaders=test_dataloader)
