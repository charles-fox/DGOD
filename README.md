# Domain Generalisation for Object Detection
This repo will have all the codes that are needed to replicate the results for our work on Domain Generalization for Object Detection.



# Datasets download and setup

We use the following four datasets.  You can download them from these links. Some require you to create an account first.

1. [Adverse Conditions Dataset with Correspondences (ACDC)](https://acdc.vision.ee.ethz.ch/download)
2. [Berkely Deep Drive 100K (BDD100K)](https://bdd-data.berkeley.edu/)
3. [Cityscapes](https://www.cityscapes-dataset.com/)
4. [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/)

Our code expects the input data format to be in csv format and we provide necessary helper functions to convert the annotations of all the above datasets into csv format. But we expect the user to download the datasets from respective websites and set up the file structure mentioned below so that the code can directly access the datasets.   Note that for BDD100K, we use the 10K split provided by the dataset for training our models. 

```
.
├── data
│   ├── BDD100k
    |     ├── images
    |     |     ├── 10k
    |     |          ├── train
    |     |          ├── val
    |     |          ├── test
    |     ├── labels
    |     |     ├── ins_seg_train.json
    |     |     ├── ins_seg_val.json
    ├──  Cityscapes
    |     ├── gtFine
    |     |     ├── train
    |     |     ├── val
    |     ├── leftImg8bit
    |     |     ├── train
    |     |     ├── val
    ├── ACDC
    |     ├── gt_detection
    |     |       ├── fog
    |     |       ├── night
    |     |       ├── rain
    |     |       ├── snow
    |     |       ├──instancesonly_test_image_info.json
    |     |       ├──instancesonly_train_image_info.json
    |     |       ├──instancesonly_val_image_info.json
    |     ├── rgb_anon
    |     |       ├── fog
    |     |       ├── night
    |     |       ├── rain
    |     |       ├── snow
    ├──  IDD
    |     ├── gtFine
    |     |     ├── train
    |     |     ├── val
    |     ├── leftImg8bit
    |     |     ├── train
    |     |     ├── val
    ├── csv_conversion.sh
    ├── Annots
```

Once the above file structure is in place, the following command needs to be executed to convert all the annotations into csv format and place them in Annots as needed by our code. 

```
cd data
source csv_conversion.sh
```

This will generate the following csv files in Annots folder where a subset of them will be used by the detector during training. 

```
1. acdc_train_all.csv
2. acdc_val_all.csv
3. bdd10k_train_all.csv
4. bdd10k_val_all.csv
5. cityscapes_train_all.csv
6. cityscapes_val_all.csv
7. idd_train_all.csv
8. idd_val_all.csv
```


# Model install and setup

We recommend to use '[Anaconda'](https://docs.anaconda.com/anaconda/install/linux/) to create a self-contained python environment for the models. Using anaconda, the following creates then activates a new environment:
```
conda create -n DGOD python
conda activate DGOD
```

Inside the environmnent, the following installs the required dependencies,
```
pip install -r requirements.txt
```

(If not using conda but you have a typical pytorch setup, you will need to install the remaining dependencies such as
```
pip3 install torchmetrics albumentations pytorch_lightning pycocotools
```
)

# Running the models

The code trains additional domain-specific classifiers for which we need the access to ground truth labels of each identified region proposal. We have made minor changes to the Faster-RCNN implementation in [WilDS](https://github.com/p-lambda/wilds/tree/main/examples/models/detection) and [Torchvision's FCOS](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py)  to obtain the ground truth labels corresponding to each instance level features. We initialize our backbone networks with ImageNet pretrained weights. We use the Pytorch-Lightning framework to train our model. 

The following are the sample commands that can be used to train the Faster R-CNN in non-dg and dg modes, respectively. 
```
python train_driving_dgfrcnn.py --exp non_dg --source_domains A  --target_domains I --weights_folder ABC2I --weights_file singlebest_a2i_frcnn 
python train_driving_dgfrcnn.py --exp dg --source_domains ABC  --target_domains I --weights_folder ABC2I --weights_file abc2i_dgfrcnn --reg_weights 0.5 0.5 0.5 0.05 0.0001
```

The following are the sample commands that can be used to train the FCOS in non-dg and dg modes, respectively. 
```
python train_driving_dgfcos.py --exp non_dg --source_domains A  --target_domains I --weights_folder ABC2I --weights_file singlebest_a2i_fcos
python train_driving_dgfcos.py --exp dg --source_domains ABC  --target_domains I --weights_folder ABC2I --weights_file abc2i_dgfcos --reg_weights 0.5 0.5 0.5 0.05 0.0001
```

It is important to note that, 'dg' mode needs more than one source domains else it might run into errors. (It is meaningless to try to learn DG features from a single source dataset).





