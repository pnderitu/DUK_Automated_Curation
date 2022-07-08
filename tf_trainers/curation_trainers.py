#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022, Paul Nderitu
# All rights reserved.
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# Tensorflow Image Curation Model Trainer

import os
import pandas
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from tensorflow.keras.layers import Input, Dropout, Dense, SeparableConv2D, Flatten, BatchNormalization
from tensorflow.keras import Model, optimizers, losses, metrics, regularizers, applications
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from datetime import datetime
from matplotlib import pyplot as plt

# Set Matplotlib lib style to seaborn
plt.style.use('seaborn-deep')


class SingleOutputClassifier:
    """
    Class to train, tune and validate a CNN model to classify images for single curation tasks.
    Uses transfer learning with the EfficientNet-B0 as the base model by default with an image input size of 224x224x3.
    Images are not flipped horizontally where laterality is unknown [Laterality and Retinal_Presence Models].
    Can train binary and multi-class models.
    """

    def __init__(self, **kwargs):
        """
        Init for the single-output classifier class.

        *** Kwargs ***

        :keyword df_path: Path to the csv file with the image dataframe
        :keyword image_path: Path to the stored images
        :keyword save_path: Path to save the model training and validation outputs
        :keyword label_column: Name of the label column
        :keyword class_dict: Dictionary of the class name and value pair
        :keyword height: Height to resize images for the model
        :keyword width: Width to resize images for the model
        :keyword batch_size: Batch size
        :keyword max_epochs: Max epochs to train for (early stopping is employed)
        :keyword hparams: Dictionary of model parameters, includes 'dropout', 'regularisation' and 'learning_rate'
        :keyword use_class_weights: Bool, if True, apply class weights (default: True)
        :keyword monitor: Metric to monitor (default: val_AUC)
        :keyword patience: Patience for early stopping (default: 3 epochs)
        """

        # Init the attributes
        self.df_path = kwargs['df_path']
        self.image_path = kwargs['image_path']
        self.save_path = kwargs['save_path']
        self.label_column = kwargs['label_column']
        self.class_dict = kwargs['class_dict']
        self.height = kwargs['image_size'][0]
        self.width = kwargs['image_size'][1]
        self.dimensions = 3
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.hparams = kwargs['hparams']
        self.monitor = 'val_' + self.label_column + '_' + 'AUC'
        self.patience = kwargs['patience']
        self.use_class_weights = kwargs['use_class_weights']
        self.save_examples = kwargs['save_examples']
        self.nclasses = len(self.class_dict.keys())
        self.base_model = applications.efficientnet.EfficientNetB0
        self.seed = kwargs['seed']

        # Pick loss function based on the class length
        if self.nclasses == 2:
            self.loss = losses.BinaryCrossentropy()
        elif self.nclasses > 2:
            self.loss = losses.CategoricalCrossentropy()

        # Add to self
        self.model_path = os.path.join(self.save_path, 'TF', 'Model', self.label_column)

        # Make folder to save the training logs
        if not os.path.exists(os.path.join(self.save_path, 'TF', 'logs', self.label_column)):
            os.makedirs(os.path.join(self.save_path, 'TF', 'logs', self.label_column))

        # Add to self
        self.logs = os.path.join(self.save_path, 'TF', 'logs', self.label_column)

        # Make folder to save the example images
        if not os.path.exists(os.path.join(self.save_path, 'TF', 'Sample_images', self.label_column)):
            os.makedirs(os.path.join(self.save_path, 'TF', 'Sample_images', self.label_column))

        # Add to self
        self.sample_images = os.path.join(self.save_path, 'TF', 'Sample_images', self.label_column)

        # Make folder to save the model results
        if not os.path.exists(os.path.join(self.save_path, 'TF', 'Results', self.label_column)):
            os.makedirs(os.path.join(self.save_path, 'TF', 'Results', self.label_column))

        # Add to self
        self.results = os.path.join(self.save_path, 'TF', 'Results', self.label_column)

        # Needed to avoid cuDNN error
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Ensure memory growth is the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        # Get the dataframe
        self._get_df()

    def _get_df(self):
        """
        Function to load the csv file to a df and select the correct subgroups
        """

        # Load the dataframe from csv
        all_img_df = pandas.read_csv(self.df_path, na_values=' ', header=0, na_filter=False)

        # Map the label_column class dict strings to integers
        all_img_df = all_img_df.replace(self.class_dict)

        # Select the correct images for gradability and retinal field models (known laterality retinal images)
        if self.label_column in ['Gradability']:
            select_img_df = all_img_df[all_img_df['Known_Laterality_Retinal'] == 'Yes']
        elif self.label_column in ['Retinal_Field']:
            select_img_df = all_img_df[all_img_df['Known_Laterality_Retinal'] == 'Yes']
        else:
            select_img_df = all_img_df
        dropped_images = all_img_df.index.size - select_img_df.index.size

        # Select images with a known label as defined in the class dictionary
        img_df = select_img_df[select_img_df[self.label_column].isin([v for v in self.class_dict.values()])]
        excluded_images = select_img_df.index.size - img_df.index.size

        # Reset index after dropping examples
        self.img_df = img_df.reset_index(drop=True)

        # Print Metrics
        print('-------------------------------------------------------------------------------------------')

        print('Running Single-output Curation Classifier')

        print('Total images = {}, Images dropped = {}, Images excluded = {}\n'.
              format(all_img_df.index.size, dropped_images, excluded_images))
        self._print_metrics(self.img_df, 'Eligible')

    def _get_class_weights(self):
        """
        Method to compute class weights.

        :return: Class weight dictionary
        """

        print('*** Using class weights ***')

        class_weight = {}

        # Scale the dataset size by the number of classes
        scaled_dataset_size = self.train_df.index.size / self.nclasses

        for c in self.class_dict:
            class_size = self.train_df[self.train_df[self.label_column] == self.class_dict[c]].index.size
            weight_c = round(((1 / class_size) * scaled_dataset_size), 4)
            print('{} class weight = {}'.format(c, weight_c))
            class_weight.update({int(self.class_dict[c]): weight_c})
        print('\n')

        return class_weight

    def _print_metrics(self, df, df_name: str):
        """
        Helper function to print the metrics of the df size, and size of each of the classes

        :param df: Pandas dataframe
        :param df_name: Dataframe name (must be 'Train', 'Val' or 'Eligible')
        """

        # Check the correct dataframe name has been used
        assert df_name in ['Train', 'Val', 'Eligible'], 'Dataframe name must be Train, Val or Eligible' \
                                                        ' got {}'.format(df_name)

        # Print Metrics
        print('-------------------------------------------------------------------------------------------')

        print('{} size = {}'.format(df_name, df.index.size))

        print('-------------------------------------------------------------------------------------------')

        for c in self.class_dict:
            print('{} images: {}'.format(c, df[df[self.label_column] == self.class_dict[c]].index.size))

        print('-------------------------------------------------------------------------------------------\n')

    @tf.function  # Enables graph execution and speeds up execution
    def _parse_image(self, data):
        """
        Helper function to parse paths to images

        :param data: Includes the path to the image file and laterality
        :return: Resized image
        """
        # Unpack
        path, laterality = data

        # Decode image
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=self.dimensions, expand_animations=False)

        # Resize image
        resized_image = tf.image.resize(image, size=(self.height, self.width))

        # If the model in not a laterality or Retinal_Presence model, flip all left eye images to right orientation
        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            if laterality == tf.constant('Left', dtype=laterality.dtype):
                resized_image = tf.image.flip_left_right(resized_image)
            elif laterality == tf.constant('Right', dtype=laterality.dtype):
                pass
            elif laterality == tf.constant('Unidentifiable', dtype=laterality.dtype):
                tf.print('Warning, unidentifiable laterality image detected, check the laterality definitions!')
                pass
            else:
                tf.print('Warning, missing laterality image detected, check the laterality definitions!')
                pass
        return resized_image

    @classmethod
    def _augmentations(cls, image_tensor):
        """
        This function applies sequential data augmentations to tf image tensors and is mapped to the train dataset only.

        :return: Augmented image
        """

        # Scale the images first
        image_tensor = image_tensor / 255

        # Apply sequential augmentations
        aug_image = tf.image.random_hue(image_tensor, 0.025)
        aug_image = tf.image.random_contrast(aug_image, 0.75, 1.5)
        aug_image = tf.image.random_saturation(aug_image, 0.75, 1.5)
        aug_image = tf.image.random_brightness(aug_image, 0.15)
        aug_image = tf.image.random_flip_up_down(aug_image)

        # Clip values between 0 and 1 after the sequential augmentations
        aug_image = tf.clip_by_value(aug_image, 0, 1)

        # Return to 0 to 255 range as EfficientNet has a rescaling layer!
        aug_image = aug_image * 255

        return aug_image

    def _example_images(self, tf_dataset, dataset_name: str):
        """
        Helper function to visualise plot of images from the train (+aug) and val datasets. By default saves the
        images into the sample_images folder. Dataset name must be train, aug_train or val.

        :param tf_dataset: Tensorflow dataset
        :param dataset_name: Dataset set name (must be 'train', 'aug_train' or 'val')
        :return: None (images saved)
        """

        # Check the correct dataset name has been used
        assert dataset_name in ['train', 'aug_train', 'val'], 'Dataset name must be train, aug_train or val, ' \
                                                              'got {}'.format(dataset_name)

        # Get a batch of images and labels
        image_batch, label_batch = next(tf_dataset.batch(self.batch_size).as_numpy_iterator())

        # Onehot encoded columns returned to integer format
        if self.nclasses > 2:
            label_batch = label_batch.argmax(axis=1)

        # Plot a batch of example images
        plt.figure(figsize=(24, 16))

        for i in range(32):
            plt.subplot(4, 8, i + 1)
            plt.imshow(image_batch[i].astype(int))
            label = label_batch[i]
            plt.title(*[c for c, v in self.class_dict.items() if v == label], size='small')
            plt.axis("off")
        plt.suptitle('{}: {} images'.format(self.label_column, dataset_name))

        # Save plot as a png
        if self.save_examples:
            plt.savefig(os.path.join(self.sample_images, f'{self.label_column}_{dataset_name}_examples.png'), dpi=300)

    def _make_dataset(self, df):
        """
        Helper function that takes a pandas DF and returns a TF dataset.
        Binary labels are label encoded (native) and the multiclass labels are one-hot encoded.

        :param df: Pandas dataframe
        :return: TF dataset
        """

        # If this is for a laterality or Retinal_Presence model then this column does not exist
        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            laterality = df['Laterality']
        else:
            laterality = None

        # Make the TF dataset as (features, label)
        if self.nclasses == 2:
            print('*** Label Encoding Applied ***')
            tf_ds = tf.data.Dataset.from_tensor_slices(((df['Path'], laterality), df[self.label_column]))
        else:
            print('*** OneHot Label Encoding Applied ***')
            tf_ds = tf.data.Dataset.from_tensor_slices(((df['Path'], laterality), tf.one_hot(df[self.label_column],
                                                                                             self.nclasses)))

        return tf_ds

    def _get_train_val(self, train_val_split: float = 0.88, make_train: bool = True, make_val: bool = True):
        """
        Function for group splitting images into a train and val datasets. Grouping is via the PID is present so that
        patients with multiple images are exclusively in the train or val dataset.
        After dataframes are created, they are converted into TF dataset objects for optimised training.
        The dataframes are saved as a csv for later use/verification. Random state seeds set to allow repeatability.

        :param train_val_split: Size of the train and val dataset (default: 0.88) which leaves 0.12 for the val set
        Assumes the test set split already at 20%.
        :param make_train: Bool, if True, make val dataset (default: True)
        :param make_val: Bool, if True, make val dataset (default: True)
        :return: Train and val datasets
        """

        # Initialise the stratified group split
        if 'PID' in self.img_df.columns:
            train_val_split = GroupShuffleSplit(n_splits=1, train_size=train_val_split, random_state=self.seed)

            # All PIDs
            all_PIDs = self.img_df['PID']

            # Get split indices for train and val datasets
            train_index, val_index = next(train_val_split.split(X=self.img_df, groups=all_PIDs))

            # Apply splits
            train_df, val_df = self.img_df.iloc[train_index], self.img_df.iloc[val_index]

        else:
            train_val_split = ShuffleSplit(n_splits=1, train_size=train_val_split, random_state=self.seed)

            # Get split indices for train and val datasets
            train_index, val_index = next(train_val_split.split(X=self.img_df))

            # Apply splits
            train_df, val_df = self.img_df.iloc[train_index], self.img_df.iloc[val_index]

        # Store the train and val_df for later use or if needed for verification
        train_df.to_csv(os.path.join(self.results, f'{self.label_column}_train_df.csv'), index=False)
        val_df.to_csv(os.path.join(self.results, f'{self.label_column}_val_df.csv'), index=False)

        # Update self but shuffle and reset the index
        self.train_df = train_df.sample(frac=1, random_state=7).reset_index(drop=True)
        self.val_df = val_df.sample(frac=1, random_state=7).reset_index(drop=True)

        # Update train and val steps
        self.train_steps = int(self.train_df.index.size / self.batch_size) + 1
        self.val_steps = int(self.val_df.index.size / self.batch_size) + 1

        # Return train if True
        if make_train:
            # Print metrics
            self._print_metrics(self.train_df, 'Train')

            # Make train dataset
            train_ds = self._make_dataset(self.train_df)

            # Map to the image decoder
            train_ds = train_ds.map(lambda x, y: (self._parse_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

            # Get sample images and save to the example images folder
            self._example_images(train_ds, 'train')

            # Cache the dataset for performance and shuffle (shuffle size = 1000)
            train_ds = train_ds.cache().shuffle(buffer_size=1000)

            # Apply the augmentations
            aug_ds = train_ds.map(lambda x, y: (self._augmentations(x), y), num_parallel_calls=tf.data.AUTOTUNE)

            # Get sample images and save to the example images folder
            self._example_images(aug_ds, 'aug_train')

            # Batch and Prefetch for performance
            aug_ds = aug_ds.batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE).repeat()

        else:
            aug_ds = None

        # Return val dataset
        if make_val:
            # Print metrics
            self._print_metrics(self.val_df, 'Val')

            # Make val dataset
            val_ds = self._make_dataset(self.val_df)

            # Map to the image decoder
            val_ds = val_ds.map(lambda x, y: (self._parse_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

            # Get sample images and save to the example images folder
            self._example_images(val_ds, 'val')

            # Cache, Batch and Prefetch
            val_ds = val_ds.cache().batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE)

        else:
            val_ds = None

        return aug_ds, val_ds

    def _build_model(self):
        """
        Function to build tensorflow model with modifiable hyperparameters.

        :return: Compiled model
        """

        if self.nclasses == 2:
            activation = tf.nn.sigmoid
            print('Using Sigmoid Activation')

            # Define the dense nodes
            dense_length = 1

        else:
            activation = tf.nn.softmax
            print('Using Softmax Activation')

            # Define the dense nodes
            dense_length = self.nclasses

        # Base Model
        Base_Model = self.base_model(input_shape=(self.height, self.width, self.dimensions), weights='imagenet',
                                     include_top=False, pooling=None)

        Base_Model.trainable = False  # Set trainable to false to freeze base model

        # Full Model
        Inputs = Input(shape=(self.height, self.width, self.dimensions))
        Features = Base_Model(Inputs, training=False)  # Set training to false to store batch norm weights
        C1 = SeparableConv2D(16, 3, activation='swish')(Features)
        B1 = BatchNormalization()(C1)
        C2 = SeparableConv2D(32, 3, activation='swish')(B1)
        B2 = BatchNormalization()(C2)
        F1 = Flatten()(B2)
        DO = Dropout(self.hparams['dropout'])(F1)
        Out = Dense(dense_length, activity_regularizer=regularizers.L2(self.hparams['regularisation']),
                    activation=activation, name=self.label_column)(DO)

        # Define the model
        model = Model(inputs=Inputs, outputs=Out)

        # Model compile and fit
        # Optimiser is RMSProp with momentum at 0.9 as per EfficientNet paper
        model.compile(loss={self.label_column: self.loss},
                      optimizer=optimizers.RMSprop(learning_rate=self.hparams['learning_rate'], momentum=0.9),
                      metrics=[metrics.AUC(name=self.label_column + '_AUC'),
                               metrics.AUC(name=self.label_column + '_PRC', curve='PR')])

        print('Loss: {}\n'.format(self.loss.name))

        return model

    def tune_model(self, tune_iterations: int = 10):
        """
        Tunes classification model parameters and saves as a log via the Tensorboard callback.

        :param tune_iterations: Number of times to randomly sample from the hyperparameter intervals (default: 10)
        """

        # Get the train and val datasets
        train_ds, val_ds = self._get_train_val()

        # Define hparams and intervals to randomly sample
        hp_dropout = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
        hp_lr = hp.HParam('learning_rate', hp.RealInterval(1e-4, 1e-2))
        hp_c = hp.HParam('regularisation', hp.RealInterval(0.0001, 0.01))

        # Random search the hyperparameters for 10 trials
        for s in range(0, tune_iterations):
            dropout_rate = hp_dropout.domain.sample_uniform()
            regularisation = hp_c.domain.sample_uniform()
            learning_rate = hp_lr.domain.sample_uniform()
            print('\n----------------------- Performing Hyperparameter Search: Run {}-----------------------\n'
                  'Trying dropout {}, learning rate {}, regularisation {}\n'.format(s, dropout_rate, learning_rate,
                                                                                    regularisation))
            # Define hparams
            hparams = {'dropout': dropout_rate, 'learning_rate': learning_rate, 'regularisation': regularisation}

            # Class weights
            if self.use_class_weights:
                class_weights = self._get_class_weights()

            else:
                class_weights = None
                print('No class weight applied\n')

            # Build the model
            model = self._build_model()

            # Fit and train for 2 epochs
            model.fit(train_ds, steps_per_epoch=self.train_steps, epochs=3, class_weight=class_weights)

            # Get performance metrics for the hparams
            loss, m_auc, m_prc = model.evaluate(val_ds, steps=self.val_steps)

            # Write metrics to tensorboard counting each run as 1 'step'
            with tf.summary.create_file_writer(os.path.join(self.logs, 'hparam_tuning/{}-{}'.format(self.label_column,
                                                                                                    s))).as_default():
                hp.hparams(hparams)
                tf.summary.scalar('loss', loss, step=1)
                tf.summary.scalar('val_AUC', m_auc, step=1)
                tf.summary.scalar('val_AUPRC', m_prc, step=1)

    def _compile_and_fit(self, train_dataset, val_dataset):
        """
        Compiles and fits models including callbacks. Saves models to path at each epoch if val AUC better than
        the last training epoch.

        :param train_dataset: Training dataset of type tf dataset with tuple (features, targets)
        :param val_dataset: Val dataset of type tf dataset with tuple (features, targets)
        """

        print('Monitoring {} for early stopping and model checkpoints'.format(self.monitor))

        # Set the correct mode depending on monitored value
        if 'AUC' in self.monitor:
            mode = 'max'
        elif 'loss' in self.monitor:
            mode = 'min'
        else:
            mode = 'auto'

        # Callbacks
        ES = EarlyStopping(monitor=self.monitor, patience=self.patience, mode=mode, restore_best_weights=True)

        MS = ModelCheckpoint(filepath=self.model_path + "-AUC{val_" + self.label_column + "_AUC:.3f}-e{epoch:02d}",
                             monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode=mode,
                             save_freq='epoch')

        TB = TensorBoard(
            log_dir=os.path.join(self.logs, f'{self.label_column}-{datetime.now().strftime("%y%m%d-%H%M%S")}'),
            profile_batch=0, write_graph=False)

        # LR Scheduler (Exponential decay) after 2 epochs
        def LR_scheduler(epoch):
            if epoch < 1:
                return self.hparams['learning_rate']
            else:
                return self.hparams['learning_rate'] * tf.math.exp(0.1 * (1 - epoch))

        LR = LearningRateScheduler(LR_scheduler, verbose=2)

        # Class weights
        if self.use_class_weights:
            class_weights = self._get_class_weights()

        else:
            class_weights = None
            print('No class weight applied\n')

        # Build the model
        model = self._build_model()

        # Fit and train
        model.fit(train_dataset, steps_per_epoch=self.train_steps, class_weight=class_weights, epochs=self.max_epochs,
                  validation_data=val_dataset, validation_steps=self.val_steps, callbacks=[ES, MS, TB, LR])

        # Fine-tune whole model
        model.trainable = True

        # Initialise new callbacks
        MS2 = ModelCheckpoint(filepath=self.model_path + "-AUC{val_" + self.label_column + "_AUC:.3f}"
                                                                                           "-e{epoch:02d}_Full",
                              monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False,
                              mode=mode, save_freq='epoch')

        TB2 = TensorBoard(
            log_dir=os.path.join(self.logs, f'{self.label_column}-{datetime.now().strftime("%y%m%d-%H%M%S")}_Full'),
            profile_batch=0, write_graph=False)

        # LR Scheduler 2 (Exponential decay) - Reduced LR by factor of 10 if fine-tuning whole model
        def LR_scheduler2(epoch):
            if epoch < 1:
                return self.hparams['learning_rate'] * 0.1
            else:
                return self.hparams['learning_rate'] * 0.1 * tf.math.exp(0.1 * (1 - epoch))

        LR2 = LearningRateScheduler(LR_scheduler2, verbose=2)

        model.fit(train_dataset, steps_per_epoch=self.train_steps, class_weight=class_weights, epochs=self.max_epochs,
                  validation_data=val_dataset, validation_steps=self.val_steps, callbacks=[ES, MS2, TB2, LR2])

    def train_val_model(self):
        """
        Main function that trains and validates the model.
        """

        # Print laterality message
        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            print('Left eye images will be flipped to right eye orientation')

        # Get the train and val datasets
        train_ds, val_ds = self._get_train_val()

        # Compile, fit and save the best models
        self._compile_and_fit(train_dataset=train_ds, val_dataset=val_ds)


class MultiOutputClassifier(SingleOutputClassifier):
    """
    This class performs multi-output training of the label column and uses the aux_column as auxiliary labels.
    """

    def __init__(self, **kwargs):

        """
        Init for multi-output trainer.

        *** Kwargs ***

        :keyword aux_column: The name of the auxiliary label column
        :keyword aux_dict: Dictionary of the auxiliary label name and value pairs
        """

        # Add the aux column and aux_dict attributes
        self.aux_column = kwargs['aux_column']
        self.aux_dict = kwargs['aux_dict']
        self.aux_nclasses = len(self.aux_dict.keys())

        # Inherit the rest of the init attributes and methods from the parent class
        super().__init__(**kwargs)

        # Monitor the val_Loss for Multi-output models instead of AUC (as there are now two AUCs!)
        self.monitor = 'val_loss'

        # Set the correct auxiliary loss
        if self.aux_nclasses == 2:
            self.aux_loss = losses.BinaryCrossentropy()
        elif self.aux_nclasses > 2:
            self.aux_loss = losses.CategoricalCrossentropy()

    def _get_df(self):
        """
        Function to load the csv file into a df and select the correct subgroups
        """

        # Load the dataframe from csv
        all_img_df = pandas.read_csv(self.df_path, na_values=' ', header=0, na_filter=False)

        # Map the label_column class dict strings to integers
        all_img_df = all_img_df.replace(self.class_dict)

        # Map the aux_column class dict strings to integers
        all_img_df = all_img_df.replace(self.aux_dict)

        # Select the correct images for gradability and retinal field models (known laterality retinal images)
        if self.label_column in ['Gradability']:
            select_img_df = all_img_df[all_img_df['Known_Laterality_Retinal'] == 'Yes']
        elif self.label_column in ['Retinal_Field']:
            select_img_df = all_img_df[all_img_df['Known_Laterality_Retinal'] == 'Yes']
        else:
            select_img_df = all_img_df
        dropped_images = all_img_df.index.size - select_img_df.index.size

        # Select images with a known label as defined in the class dictionary
        img_df = select_img_df[select_img_df[self.label_column].isin([v for v in self.class_dict.values()])]
        select_img_df = select_img_df[select_img_df[self.aux_column].isin([v for v in self.aux_dict.values()])]
        excluded_images = select_img_df.index.size - img_df.index.size

        # Reset index after dropping examples
        self.img_df = img_df.reset_index(drop=True)

        # Print Metrics
        print('-------------------------------------------------------------------------------------------')

        print('Running Multi-output Classifier')

        print('Total images = {}, Images dropped = {}, Images excluded = {}\n'.
              format(all_img_df.index.size, dropped_images, excluded_images))
        self._print_metrics(self.img_df, 'Eligible')

    def _print_metrics(self, df, df_name: str):
        """
        Helper function to print the metrics of the df size, and size of each of the classes

        :param df: Pandas dataframe
        :param df_name: Dataframe name (must be 'Train', 'Val' , 'Eligible')
        """

        # Check the correct dataframe name has been used
        assert df_name in ['Train', 'Val', 'Eligible'], 'Dataframe name must be Train, Val or Eligible' \
                                                        ' got {}'.format(df_name)

        # Print Metrics
        print('-------------------------------------------------------------------------------------------')

        print('{} size = {}'.format(df_name, df.index.size))

        print('-------------------------------------------------------------------------------------------')

        for c in self.class_dict:
            print('Primary: {} images: {}'.format(c, df[df[self.label_column] == self.class_dict[c]].index.size))

        print('-------------------------------------------------------------------------------------------\n')

        for c in self.aux_dict:
            print('Aux: {} images: {}'.format(c, df[df[self.aux_column] == self.aux_dict[c]].index.size))

        print('-------------------------------------------------------------------------------------------\n')

    def _make_dataset(self, df):
        """
        Helper function that takes a pandas DF and returns a TF dataset.
        Binary labels are label encoded (native) and the multiclass labels are one-hot encoded.

        :param df: pandas dataframe
        :return: TF dataset
        """

        # If this is for a laterality or Retinal_Presence model then this column does not exist
        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            if self.aux_column not in ['Laterality', 'Retinal_Presence']:
                laterality = df['Laterality']
            else:
                laterality = None
        else:
            laterality = None

        # Make the TF dataset
        if self.nclasses == 2 and self.aux_nclasses == 2:
            print('*** Label Encoding Applied to {} and {}*** '.format(self.label_column, self.aux_column))
            tf_ds = tf.data.Dataset.from_tensor_slices(((df['Path'], laterality), {self.label_column:
                                                                                       df[self.label_column],
                                                                                   self.aux_column:
                                                                                       df[self.aux_column]}))

        elif self.nclasses == 2 and self.aux_nclasses > 2:
            print('*** Label Encoding Applied to {} and OneHot Encoding Applied to {} ***'.format(
                self.label_column, self.aux_column))
            tf_ds = tf.data.Dataset.from_tensor_slices(((df['Path'], laterality), {self.label_column:
                                                                                       df[self.label_column],
                                                                                   self.aux_column:
                                                                                       tf.one_hot(df[self.aux_column],
                                                                                                  self.aux_nclasses)}))

        elif self.nclasses > 2 and self.aux_nclasses == 2:
            print('*** OneHot Encoding Applied to {} and Label Encoding Applied to {} ***'.format(
                self.label_column, self.aux_column))
            tf_ds = tf.data.Dataset.from_tensor_slices(((df['Path'], laterality),
                                                        {self.label_column:
                                                             tf.one_hot(df[self.label_column],
                                                                        self.nclasses),
                                                         self.aux_column: df[self.aux_column]}))

        else:
            print('*** OneHot Encoding Applied to {} and {} ***'.format(
                self.label_column, self.aux_column))
            tf_ds = tf.data.Dataset.from_tensor_slices(((df['Path'], laterality),
                                                        {self.label_column:
                                                             tf.one_hot(df[self.label_column],
                                                                        self.nclasses),
                                                         self.aux_column:
                                                             tf.one_hot(df[self.aux_column],
                                                                        self.aux_nclasses)}))

        return tf_ds

    def _example_images(self, tf_dataset, dataset_name: str):
        """
        Helper function to visualise plot of images from the train (+aug) and val datasets. By default saves the
        images into the sample_images folder. Dataset name must be train, aug_train or val.
        Saves augmented images if dataset name is train.

        :param tf_dataset: Tensorflow dataset
        :param dataset_name: Dataset set name (must be 'train' or 'val')
        :return: None (images saved)
        """

        # Check the correct dataset name has been used
        assert dataset_name in ['train', 'aug_train', 'val'], 'Dataset name must be train, aug_train or val ' \
                                                              'got {}'.format(dataset_name)

        # Get a batch of images and labels
        image_batch, label_batch = next(tf_dataset.batch(32).as_numpy_iterator())

        # Onehot encoded columns must be returned to integer format
        if self.nclasses > 2:
            primary_label_batch = label_batch[self.label_column].argmax(axis=1)
        else:
            primary_label_batch = label_batch[self.label_column]

        if self.aux_nclasses > 2:
            aux_label_batch = label_batch[self.aux_column].argmax(axis=1)
        else:
            aux_label_batch = label_batch[self.aux_column]

        # Plot a batch of example images
        plt.figure(figsize=(24, 16))

        for i in range(32):
            plt.subplot(4, 8, i + 1)
            plt.imshow(image_batch[i].astype(int))
            label_primary = primary_label_batch[i]
            label_aux = aux_label_batch[i]
            plt.title('{}-{}'.format(*[p for p, v in self.class_dict.items() if v == label_primary],
                                     *[a for a, v in self.aux_dict.items() if v == label_aux]), size='small')
            plt.axis("off")
        plt.suptitle('{}-{}: {} images'.format(self.label_column, self.aux_column, dataset_name))

        # Save plot as a png
        if self.save_examples:
            plt.savefig(os.path.join(self.sample_images,
                                      f'{self.label_column}_{self.aux_column}_{dataset_name}_examples.png'), dpi=300)

    def _build_model(self):
        """
        Function to build tensorflow model with modifiable hyperparameters.

        :param: hparams: Dictionary of param key and value
        :param: print_save_model_summary: Bool, print and save model summary if True (default: False)
        :return: Compiled model
        """

        if self.nclasses == 2:
            primary_activation = tf.nn.sigmoid
            print('Using Sigmoid Activation for {} Classification'.format(self.label_column))

            # Define the dense nodes
            primary_dense_length = 1

        else:
            primary_activation = tf.nn.softmax
            print('Using Softmax Activation for {} Classification'.format(self.label_column))

            # Define the dense nodes
            primary_dense_length = self.nclasses

        if self.aux_nclasses == 2:
            aux_activation = tf.nn.sigmoid
            print('Using Sigmoid Activation for {} Classification'.format(self.aux_column))

            # Define the dense nodes
            aux_dense_length = 1

        else:
            aux_activation = tf.nn.softmax
            print('Using Softmax Activation for {} Classification'.format(self.aux_column))

            # Define the dense nodes
            aux_dense_length = self.aux_nclasses

        # Base Model
        Base_Model = self.base_model(input_shape=(self.height, self.width, self.dimensions), weights='imagenet',
                                     include_top=False, pooling=None)

        Base_Model.trainable = False  # Set trainable to false to freeze base model

        # Full Model
        Inputs = Input(shape=(self.height, self.width, self.dimensions))
        Features = Base_Model(Inputs, training=False)  # Set training to false to store batch norm weights
        C1 = SeparableConv2D(16, 3, activation='swish')(Features)
        B1 = BatchNormalization()(C1)
        C2 = SeparableConv2D(32, 3, activation='swish')(B1)
        B2 = BatchNormalization()(C2)
        F1 = Flatten()(B2)
        DO = Dropout(self.hparams['dropout'])(F1)
        Primary = Dense(primary_dense_length, activity_regularizer=regularizers.L2(self.hparams['regularisation']),
                        activation=primary_activation, name=self.label_column)(DO)
        Aux = Dense(aux_dense_length, activity_regularizer=regularizers.L2(self.hparams['regularisation']),
                    activation=aux_activation, name=self.aux_column)(DO)

        # Define the model
        model = Model(inputs=Inputs, outputs=[Primary, Aux])

        # Define loss weighting (default 1:1)
        loss_weights = {self.label_column: 1.0, self.aux_column: 1.0}

        # Model compile and fit
        # Optimiser is RMSProp with momentum at 0.9 as per EfficientNet paper
        model.compile(loss={self.label_column: self.loss,
                            self.aux_column: self.aux_loss},
                      loss_weights=loss_weights,
                      optimizer=optimizers.RMSprop(learning_rate=self.hparams['learning_rate'], momentum=0.9),
                      metrics=[metrics.AUC(name='AUC'), metrics.AUC(name='PRC', curve='PR')])

        print('Primary Loss: {}\nAux Loss: {}\n'.format(self.loss.name, self.aux_loss.name))

        return model

    def _compile_and_fit(self, train_dataset, val_dataset):
        """
        Compiles and fits models including callbacks. Saves models to path at each epoch if (total) val_loss lower than
        the last training epoch.

        :param train_dataset: Training dataset of type tf dataset with tuple (features, targets)
        :param val_dataset: Val dataset of type tf dataset with tuple (features, targets)
        """

        print('Monitoring {} for early stopping and model checkpoints'.format(self.monitor))

        # Set the correct mode depending on monitored value
        if 'AUC' in self.monitor:
            mode = 'max'
        elif 'loss' in self.monitor:
            mode = 'min'
        else:
            mode = 'auto'

        # Callbacks
        ES = EarlyStopping(monitor=self.monitor, patience=self.patience, mode=mode, restore_best_weights=True)

        MS = ModelCheckpoint(filepath=self.model_path + "-AUC{val_" + self.label_column + "_AUC:.3f}-" +
                                                   self.aux_column + "-AUC{val_" + self.aux_column +
                                      "_AUC:.3f}-e{epoch:02d}",
                             monitor=self.monitor, verbose=1, save_best_only=True, save_weights_only=False, mode=mode,
                             save_freq='epoch')

        TB = TensorBoard(
            log_dir=os.path.join(self.logs, f'{self.label_column}-'
                                            f'{self.aux_column}-{datetime.now().strftime("%y%m%d-%H%M%S")}'),
            profile_batch=0, write_graph=False)

        # LR Scheduler (Exponential decay)
        def LR_scheduler(epoch):
            if epoch < 1:
                return self.hparams['learning_rate']
            else:
                return self.hparams['learning_rate'] * tf.math.exp(0.1 * (1 - epoch))

        LR = LearningRateScheduler(LR_scheduler, verbose=2)

        # No Class weights
        class_weights = None
        print('No class weight applied\n')

        # Build the model
        model = self._build_model()

        # Fit and train
        model.fit(train_dataset, steps_per_epoch=self.train_steps, class_weight=class_weights, epochs=self.max_epochs,
                  validation_steps=self.val_steps, validation_data=val_dataset, callbacks=[ES, MS, TB, LR])

        # Fine-tune whole model
        model.trainable = True

        # Initialise new callbacks
        MS2 = ModelCheckpoint(filepath=self.model_path + "-AUC{val_" + self.label_column + "_AUC:.3f}-" +
                                       self.aux_column + "-AUC{val_" + self.aux_column + "_AUC:.3f}-e{epoch:02d}_Full",
                              monitor=self.monitor, verbose=1, save_best_only=True,
                              save_weights_only=False, mode=mode, save_freq='epoch')

        TB2 = TensorBoard(log_dir=os.path.join(self.logs, f'{self.label_column}-{self.aux_column}-'
                                                          f'{datetime.now().strftime("%y%m%d-%H%M%S")}_Full'),
                          profile_batch=0, write_graph=False)

        # LR Scheduler 2 (Exponential decay) - Reduced LR by factor of 10 if fine-tuning whole model
        def LR_scheduler2(epoch):
            if epoch < 1:
                return self.hparams['learning_rate'] * 0.1
            else:
                return self.hparams['learning_rate'] * 0.1 * tf.math.exp(0.1 * (1 - epoch))

        LR2 = LearningRateScheduler(LR_scheduler2, verbose=2)

        model.fit(train_dataset, steps_per_epoch=self.train_steps, class_weight=class_weights, epochs=self.max_epochs,
                  validation_steps=self.val_steps, validation_data=val_dataset, callbacks=[ES, MS2, TB2, LR2])

    def tune_model(self, tune_iterations: int = 10):
        """
        Tunes classification model parameters and saves as a log via the Tensorboard callback.

        :param tune_iterations: Number of times to randomly sample from the hyperparameter intervals (default: 10)
        """

        # Get the train and val datasets
        train_ds, val_ds = self._get_train_val()

        # Define hparams and intervals to randomly sample
        hp_dropout = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
        hp_lr = hp.HParam('learning_rate', hp.RealInterval(1e-4, 1e-2))
        hp_c = hp.HParam('regularisation', hp.RealInterval(0.0001, 0.01))

        # Random search the hyperparameters for 10 trials
        for s in range(0, tune_iterations):
            dropout_rate = hp_dropout.domain.sample_uniform()
            regularisation = hp_c.domain.sample_uniform()
            learning_rate = hp_lr.domain.sample_uniform()
            print('\n----------------------- Performing Hyperparameter Search: Run {}-----------------------\n'
                  'Trying dropout {}, learning rate {}, regularisation {}\n'.format(s, dropout_rate, learning_rate,
                                                                                    regularisation))
            # Define hparams
            hparams = {'dropout': dropout_rate, 'learning_rate': learning_rate, 'regularisation': regularisation}

            # No class weights
            class_weights = None
            print('No class weight applied\n')

            # Build the model
            model = self._build_model()

            # Fit and train for 2 epochs
            model.fit(train_ds, steps_per_epoch=self.train_steps, class_weight=class_weights, epochs=3)

            # Get performance metrics for the hparams
            total_loss, primary_loss, aux_loss, primary_auc, aux_auc, primary_prc, aux_prc = model.evaluate(
                val_ds, steps=self.val_steps)

            # Write metrics to tensorboard counting each run as 1 'step'
            with tf.summary.create_file_writer(os.path.join(self.logs,
                                                            'hparam_tuning/{}-{}-Multi-{}'.format(self.label_column,
                                                                                                  self.aux_column,
                                                                                                  s))).as_default():
                hp.hparams(hparams)
                tf.summary.scalar('total_loss', total_loss, step=1)
                tf.summary.scalar('primary_loss', primary_loss, step=1)
                tf.summary.scalar('aux_loss', aux_loss, step=1)
                tf.summary.scalar('primary_AUC', primary_auc, step=1)
                tf.summary.scalar('aux_AUC', aux_auc, step=1)
                tf.summary.scalar('primary_AUPRC', primary_prc, step=1)
                tf.summary.scalar('aux_AUPRC', aux_prc, step=1)
