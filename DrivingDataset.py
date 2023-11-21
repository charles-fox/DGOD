import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

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

    def __len__(self):    #l
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
