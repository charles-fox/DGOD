# Domain Generalisation for Object Detection
This repo will have all the codes that are needed to replicate the results for our work on Domain Generalization for Object Detection.

We used the following datasets for our experiments. 

1. [Adverse Conditions Dataset with Correspondences (ACDC)](https://acdc.vision.ee.ethz.ch/download)
2. [Berkely Deep Drive 100K (BDD100K)](https://bdd-data.berkeley.edu/)
3. [Cityscapes](https://www.cityscapes-dataset.com/)
4. [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/)

Our code expects the input data format to be in csv format and we provide necessary helper functions to convert the annotations of all the above datasets into csv format. But we expect the users to download the datasets from respective websites and follow the file structure mentioned below so that the code can directly access the datasets. The file requirements.txt has all the dependencies needed to run this code. Note that for BDD100K, we use the 10K split provided by the dataset for training our models. 

# Directory structure for datasets

```
.
├── datasets
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
    ├── to_csv_conversion.sh
    ├── Annots
```

Once above directory structure is ensured, the following command needs to be executed to convert all the annotations into csv format and place them in Annots as needed by our code. 

```
./to_csv_conversion.sh
```

The above command will generate the following csv files in Annots folder where a subset of them will be used by the detector during training. 

```
1. acdc_train.csv
2. acdc_val.csv
3. bdd10k_train_all.csv
4. bdd10k_val_all.csv
5. cityscapes_clear_all_train.csv
6. cityscapes_clear_all_val.csv
7. idd_train.csv
8. idd_val.csv
```


# Training the Faster R-CNN

We recommend the users to use '[Anaconda'](https://docs.anaconda.com/anaconda/install/linux/) to create a virtual environment. The following command can be used to create a new environment needed for replicating the results in this paper. 
```
conda create -n DGOD python
```

Once the environment is succesfully created, it needs to be activated using the following command. 
```
conda activate DGOD
```

We recommend to use the following command to install all the dependencies inside the DGFRCNN environment
```
pip install -r requirements.txt
```

In this code, we need to train additional domain specific classifiers for which we need the access to ground truth labels of each identified region proposal. We have made minor changes to the Faster-RCNN implementation in [WilDS](https://github.com/p-lambda/wilds/tree/main/examples/models/detection) to obtain the ground truth labels of each region proposal. We initialise our ResNet backbone with COCO pretrained weights. We use the Pytorch-Lightning framework to train our model. 

