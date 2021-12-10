# Automated Image Curation in Diabetic Retinopathy Screening using Deep Learning
This repository contains a Tensorflow 2.7.0 implementation from the paper
**['Automated Image Curation in Diabetic Retinopathy Screening using Deep Learning']().**

## Key Dependencies
   - Python v3.8.x
   - Tensorflow v2.7.x
   - Sklearn v1.0.x
   - Numpy
   - Matplotlib & Seaborn

## Modules and Resources
*  Class to tune, train and validate a single or multi-output Tensorflow classifier.
*  Class to test and get predictions from a single or multi-output Tensorflow classifier.
*  Ground truth labels for the external validation dataset (in *ground_truth/external_test_labels.csv*).

## Ground Truth Definitions
Labels for laterality, retinal presence, retinal field and gradability were defined by an ophthalmology fellow.
The figure below shows examples from the external test set. More details can be found in our paper [supplementary materials]().

![fig](ground_truth/gt_definitions.jpg )

## Use
All modes require a csv file. If training or tuning then the supplied csv is assumed to be the 
development dataset and will be randomly split 80/20 into train/val dataset with respect to the PID if provided.

**If training, tuning or testing ensure the following columns are provided and named as follows.**
- Image_ID
- Laterality 
- Retinal_Presence
- Retinal_Field 
- Gradability 
- Known_Laterality_Retinal (*For Retinal_Field and Gradability models*)
   - With 'Yes' to indicate retinal images of known laterality
- PID (*optional unique identifier column if multiple samples from the same patient*)


**If predicting then the following columns must be present**
- Image_ID
- Laterality (*For Retinal_Field and Gradability models*)
- Known_Laterality_Retinal (*For Retinal_Field and Gradability models*)

### Modes
**tune**: Tune mode will load the development csv, split into train/val datasets at 80/20 splits and train a single 
or multi-output model based on the provided label_column +/- aux-column with 20 iterations performed 
(max 5 epochs per iteration), and the hparam space is randomly sampled as follows: 
dropout (0.2, 0.5), learning_rate (1e-4, 1e-2) and regularisation (0.0001, 0.01).

**train**: Training mode will load the csv, split into train/val datasets at 80/20 splits and train a single 
or multi-output model based on the provided label_column +/- aux_column and save the model at each epoch if the 
monitored metric improves. Note, initially the encoder weights (EfficientNet-V1-B0) are frozen until early stop criteria 
are reached then the whole model is trained.

**test**: Training mode will load the csv and return confusion matrices and ROC curves for the label_column +/- 
aux_column and save the resulting figures.

**Predict**: Predict mode will load the csv and return predictions for binary labels or the argmax index for 
multi-class labels saved in the original csv.

### Single-output model with tune, train, test or predict modes
```main.py  -dp ['path to csv'] -ip ['path to images'] -sp ['path to save logs/models/examples/results'] -l ['label column e.g., Laterality'] -mt single-output -m ['mode']```

If performing inference (test or predict modes) a path to the trained model (-mp) must also be provided.

### Multi-output model with tune, train, test or predict modes
```main.py  -dp ['path to csv'] -ip ['path to images'] -sp ['path to save logs/models/examples/results'] -l ['label column e.g., Laterality']  -al ['second label column e.g., Retinal_Presence'] -mt multi-output -m ['mode']```

If performing inference (test or predict modes) a path to the trained model (-mp) must also be provided.

#### Other Arguments
- -ucw: If class weights should be used (single-output models only) (default: True)
- -bs: Batch size (default: 32)
- -do: Dropout (default: 0.2)
- -lr: Learning rate (default: 0.001)
- -r: Regularisation (default: 0.001)

#### Defaults
- *save_examples: Save a batch of image examples for inspection (default:True)*
- *seed: Random seed for splitting development dataset into train/val partitions (default: 7)* 
- *image_size: Image size (default: [224, 224])*
- *patience: Early stopping patience (default: 3 epochs)*
- *early_stopping: This is performed by default and the val_AUC is monitored for single-output models whilst the 
  val_loss is used for multi-output models*
- *train_val_split: Ratio of development dataset to split into train and val partitions (default: 80/20)*  
- *max_epochs: Maximum training epochs (default: 50 epochs)*

## Funding
This work is wholly funded by Diabetes UK via the [Sir George Alberti Training Fellowship](https://www.diabetes.org.uk/research/our-research-projects/london/nderitu-ai-retinopathy) to grant to Paul Nderitu.

## Citation
If you use this work as part of your project, please cite [Nderitu, P *et al.,* (2022)]()

```bibtex
@article{nderitup2022,
  title={Automated Image Curation in Diabetic Retinopathy Screening using Deep Learning},
  author={},
  journal={},
 year = {2022},
  number = {},
  pages = {},
  publisher = {},
  volume = {},
  doi = {},
}