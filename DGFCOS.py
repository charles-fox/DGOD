from DGcommon import *
    
class InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()

        self.classifer=nn.Linear(128,self.num_domains)

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=torch.sigmoid(self.classifer(x))
        return x

class InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_cls)
        
    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))
        return x

class InsCls(nn.Module):
    def __init__(self, num_cls):
        super(InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(256, 128)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(128, 64)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(64,self.num_cls)

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))
        return x

      
      
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    #if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        #_log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss                    
  

 

class DGFCOS(pytorch_lightning.core.module.LightningModule):
    def __init__(self,n_classes, batch_size, exp, reg_weights, tr_dataset, tr_datasets):
        super(DGFCOS, self).__init__()
        self.tr_dataset=tr_dataset
        self.tr_datasets=tr_datasets
        self.n_classes = n_classes
        self.n_domains = len(self.tr_datasets)
        self.batch_size = batch_size
        self.exp = exp
        self.reg_weights = reg_weights
      
        self.detector = fcos.fcos_resnet50_fpn(min_size=600, max_size=1200, num_classes=self.n_classes, trainable_backbone_layers=3)
          
        self.ImageDA = ImageDA(self.n_domains)
        self.InsDA = InstanceDA(self.n_domains)       
        self.InsCls = nn.ModuleList([InsCls(self.n_classes) for i in range(self.n_domains)])
        self.InsClsPrime = nn.ModuleList([InsClsPrime(self.n_classes) for i in range(self.n_domains)])
    
        self.best_val_acc = 0
        self.log('val_acc', self.best_val_acc)
        self.metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds = [0.5])   

        self.base_lr = 1e-4 
        self.momentum = 0.9
        self.weight_decay=0.0001
        
        self.detector.backbone.body.register_forward_hook(self.store_backbone_out)        
        self.detector.head.register_forward_hook(self.store_head_input)  #For instance level features
        
        self.mode = 0
        self.sub_mode = 0
    
      
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output['2']  #Output is a dict at three levels. We consider the top level feature as representative of image level feature
    
    def store_head_input(self, module, input1, output):
      temp0 = torch.reshape(input1[0][0], (input1[0][0].shape[0], input1[0][0].shape[1], input1[0][0].shape[2]*input1[0][0].shape[3]))
      temp1 = torch.reshape(input1[0][1], (input1[0][1].shape[0], input1[0][1].shape[1], input1[0][1].shape[2]*input1[0][1].shape[3]))
      temp2 = torch.reshape(input1[0][2], (input1[0][2].shape[0], input1[0][2].shape[1], input1[0][2].shape[2]*input1[0][2].shape[3]))
      temp3 = torch.reshape(input1[0][3], (input1[0][3].shape[0], input1[0][3].shape[1], input1[0][3].shape[2]*input1[0][3].shape[3]))
      temp4 = torch.reshape(input1[0][4], (input1[0][4].shape[0], input1[0][4].shape[1], input1[0][4].shape[2]*input1[0][4].shape[3]))
      self.ins_feat = torch.cat((temp0, temp1, temp2, temp3, temp4), -1).permute(0, 2, 1)
      
    def forward(self, imgs,targets=None):
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      optimizer = torch.optim.Adam([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
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
          sample_indices.append(index)  
        if(self.exp == 'dg'):
          for index in batch:		   
            sample_indices.append(index)
      return torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.batch_size, sampler=sample_indices, shuffle=False, collate_fn=collate_fn, num_workers=16)      
          
    def training_step(self, batch, batch_idx):
      imgs = list(image.cuda() for image in batch[0])         #these lines are same as FRCNN 
      targets = []
      for boxes, labels in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = labels.long().cuda()
        targets.append(target)

      # fasterrcnn takes both images and targets for training, returns
      #Detection using source images
      
      if self.mode == 0:
        loss_dict = self.detector(imgs, targets)    #diff fm FRCNN
        loss = loss_dict['classification'] + loss_dict['bbox_regression'] + loss_dict['bbox_ctrness'] #diff from FRCNN

        if(self.exp == 'dg'):             #these are same as FRCNN
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
        
        temp_loss = []    #added from FRCNN

        _ = self.detector(imgs, targets)
        ImgDA_scores = self.ImageDA(self.base_feat)
        loss_dict['DA_img_loss'] = self.reg_weights[0]*torch.nn.functional.cross_entropy(ImgDA_scores, batch[3])
        IDA_out = self.InsDA(self.ins_feat)	
        loss_dict['DA_ins_loss'] = self.reg_weights[1]*torch.nn.functional.cross_entropy(IDA_out.permute(0, 2, 1), batch[3].unsqueeze(dim=-1).repeat(1, IDA_out.shape[1]).long())
        loss_dict['Cst_loss'] = self.reg_weights[2]*torch.nn.functional.mse_loss(ImgDA_scores.unsqueeze(dim=1).repeat(1, IDA_out.shape[1], 1), IDA_out)
        loss = sum(loss for loss in loss_dict.values())
        self.mode = 0
              
      elif(self.mode == 2): #Without recording the gradients for detector, we need to update the weights for classifier weights
        loss_dict = {}
        loss = []
        for index in range(self.n_domains):
          for param in self.InsCls[index].parameters(): param.requires_grad = True
        for index in range(len(imgs)):
          with torch.no_grad():
            temp_dict = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsCls[batch[3][index].item()](self.ins_feat)
          loss.append(torch.nn.functional.cross_entropy(cls_scores, temp_dict['gt_classes'])) 
        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())
        self.mode = 0

      elif(self.mode == 3): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
        for index in range(len(imgs)):
          temp_dict = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsClsPrime[batch[3][index].item()](self.ins_feat)
          loss.append(torch.nn.functional.cross_entropy(cls_scores, temp_dict['gt_classes']))
        loss_dict['cls_prime'] = self.reg_weights[3]*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())
        self.mode = 0
        
      else: #For Mode 4
        loss_dict = {}
        loss = []
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        for index in range(len(imgs)):
          temp_dict = self.detector([imgs[index]], [targets[index]])
          for i in range(len(self.InsCls)):
            if(i != batch[3][index].item()):
              cls_scores = self.InsCls[i](self.ins_feat)    
              loss.append(torch.nn.functional.cross_entropy(cls_scores, temp_dict['gt_classes']))
        loss_dict['cls'] = self.reg_weights[4]*(torch.mean(torch.stack(loss))) # + torch.mean(torch.stack(consis_loss)))
        loss = sum(loss for loss in loss_dict.values())
        self.mode = 0
        self.sub_mode = 0
 	 
      return {"loss": loss}#, "log": torch.stack(temp_loss).detach().cpu()}


    #SAME
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
   
    #SAME
    def on_validation_epoch_end(self):
      metrics = self.metric.compute()
      self.log('val_acc', metrics['map_50'])
      print(metrics['map_per_class'], metrics['map_50'])
      self.metric.reset() 
