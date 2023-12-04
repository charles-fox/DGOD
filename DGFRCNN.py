from DGcommon import *
import fasterrcnn

class InstanceDA(torch.nn.Module):
    def __init__(self, num_domains):
        super(InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(1024, 512)     
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)    #different from FCOS
        self.dc_relu2 = nn.ReLU()            #different from FCOS
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_domains)

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)			#different from FCOS
        x=torch.sigmoid(self.classifer(x))      #different from FCOS  
        return x

class InsClsPrime(torch.nn.Module):
    def __init__(self, num_cls):
        super(InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)    #different sizes from FCOS
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)    #different sizes from FCOS
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)   #different sizes from FCOS

    def forward(self,x):     #same as FCOS
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))
        return x

class InsCls(torch.nn.Module):
    def __init__(self, num_cls):
        super(InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)      #different sizes from FCOS
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)       #different sizes from FCOS
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)   #different sizes from FCOS

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))
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
        self.ImageDA = ImageDAFPN(256, self.num_domains)
        self.InsDA = InstanceDA(self.num_domains)       
        self.InsCls = nn.ModuleList([InsCls(n_classes) for i in range(self.num_domains)])
        self.InsClsPrime = nn.ModuleList([InsClsPrime(n_classes) for i in range(self.num_domains)])
        
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



