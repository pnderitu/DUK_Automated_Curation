#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022, Paul Nderitu
# All rights reserved.
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# Tensorflow Image Curation Prediction

import os
import pandas
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from matplotlib import pyplot as plt

# Set Matplotlib lib style to seaborn
plt.style.use('seaborn-deep')


# noinspection PyBroadException
class SingleOutputPredictor:
    """
    This class takes a trained TF model and classifies images for a given label and returns a CSV file with the class
    predictions added as new column. For binary classes the prediction score is returned but for multi-class
    predictions the argmax integer class label is returned. Can also perform model testing with CM and ROC curves
    plotted.
    """

    def __init__(self, **kwargs):
        """
        Init for prediction class.

        *** Kwargs ***

        :keyword df_path: Path to the csv file with the image dataframe
        :keyword image_path: Path to the stored images
        :keyword save_path: Path to save the model training and validation outputs
        :keyword label_column: Name to give to the label column
        :keyword results_path: Results path was defined in the input for the BaseImageClassifier
        :keyword height: Height to resize images for the model
        :keyword width: Width to resize images for the model
        :keyword batch_size: Batch size
        :keyword class_dict: Dictionary of the class name and value pair
        """

        self.df_path = kwargs['df_path']
        self.image_path = kwargs['image_path']
        self.save_path = kwargs['save_path']
        self.label_column = kwargs['label_column']
        self.class_dict = kwargs['class_dict']
        self.model_path = kwargs['model_path']
        self.batch_size = kwargs['batch_size']
        self.height = kwargs['image_size'][0]
        self.width = kwargs['image_size'][1]
        self.dimensions = 3
        self.nclasses = len(self.class_dict.keys())

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

        :return: Pandas dataframe
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

        # Reset index after dropping examples
        self.img_df = select_img_df.reset_index(drop=True)

        # Print Metrics
        print('Total images = {}, Images dropped = {}\n'.format(self.img_df.index.size, dropped_images))

    @tf.function  # Enables graph execution and speeds up execution
    def _parse_image(self, path, laterality):
        """
        Helper function to parse paths to images

        :param path: Path to the image file
        :param laterality: Image laterality
        :return: Resized image
        """

        # Decode image
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=self.dimensions, expand_animations=False)

        # Resize image
        resized_image = tf.image.resize(image, size=(self.height, self.width))

        # If model in not a laterality or Retinal_Presence trainer then flip all left eye images to right orientation
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

    def _make_dataset(self, df):
        """
        Helper function that takes a pandas DF and returns a TF dataset

        :param df: pandas dataframe
        :return: TF dataset
        """

        # If this is for a laterality or Retinal_Presence model then don't use this column
        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            laterality = df['Laterality']
        else:
            laterality = None

        # Make tf dataset
        tf_ds = tf.data.Dataset.from_tensor_slices((df['Path'], laterality))

        return tf_ds

    def get_predictions(self):
        """
        Main function that returns model predictions for the main labels.
        """

        print('Getting Predictions')

        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            print('Left eye images will be flipped to right eye orientation')

        # Load the model
        model = models.load_model(self.model_path)

        # Make test dataset
        pred_ds = self._make_dataset(self.img_df)

        # Map to the image decoder
        pred_ds = pred_ds.map(lambda x, y: self._parse_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        pred_ds = pred_ds.batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Predict
        predictions = model.predict(pred_ds, verbose=1)

        # If binary return the prediction probability of the +ve class otherwise return the argmax
        if predictions.shape[1] == 1:
            predictions = predictions

        else:
            predictions = predictions.argmax(axis=1)

        # Put predictions in a dataframe
        predictions_df = pandas.DataFrame(data=[*predictions], columns=[f'{self.label_column}_P'])

        # Add the predictions to the path only dataframe by concat
        combined_df = pandas.concat([self.img_df, predictions_df], axis=1)

        # Save predictions
        combined_df.to_csv(os.path.join(self.results, f'{self.label_column}_Predictions_DF.csv'), index=False)

    def test_model(self):
        """
        Main function that tests the model and saves the confusion matrix, ROC and PRC per class
        """

        # Get the test datasets
        test_ds = self._make_dataset(self.img_df)

        # Map to the image decoder
        test_ds = test_ds.map(lambda x, y: self._parse_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        test_ds = test_ds.batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Get labels as one hot encoded values to plot per class outcomes
        labels = tf.one_hot(self.img_df[self.label_column], self.nclasses).numpy()

        # Load the model
        model = models.load_model(self.model_path)

        # Compute test steps
        test_steps = int(self.img_df.index.size / self.batch_size) + 1

        # Use the model to predict the values from the validation dataset for the label column
        predictions = model.predict(test_ds, steps=test_steps, verbose=1)

        # Get the predictions of both the positive and negative class if binary to enable per class metrics
        if self.nclasses == 2:
            predictions = np.column_stack((1 - predictions, predictions))
        else:
            predictions = predictions

        # Calculate the confusion matrix
        cm = tf.math.confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1),
                                      num_classes=self.nclasses).numpy()

        # CM plot
        figure, (ax_cm, ax_roc, ax_prc) = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

        cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_dict.keys())
        cm_plot.plot(ax=ax_cm)
        cm_plot.figure_.set_label('Count')
        ax_cm.set_title('Confusion Matrix: {}'.format(self.label_column))

        # Plot ROC and PRC per class
        for i, c in enumerate(self.class_dict.keys()):
            fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
            class_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, linewidth=2, linestyle='--', label='{0}: {1:.3f}'.format(c, class_auc))

            p, r, _ = precision_recall_curve(labels[:, i], predictions[:, i])
            # Fixes the PR curve at 0 precision then recall is 1 (i.e. if the threshold value is 0).
            p = np.insert(p, 0, 0)
            r = np.insert(r, 0, 1)
            class_pr_auc = auc(r, p)
            ax_prc.plot(r, p, linewidth=2, linestyle='--', label='{0}: {1:.3f}'.format(c, class_pr_auc))

        ax_roc.legend(loc="lower right", title="AUROC", fancybox=True)
        ax_roc.set_title('ROC: {} Classifier'.format(self.label_column))
        ax_roc.set_xlabel('1 - Specificity')
        ax_roc.set_ylabel('Sensitivity')

        ax_prc.legend(loc="lower left", title="AUPRC", fancybox=True)
        ax_prc.set_title('PRC: {} Classifier'.format(self.label_column))
        ax_prc.set_xlabel('Recall [Sensitivity]')
        ax_prc.set_ylabel('Precision')

        figure.suptitle('Confusion Matrix, AUROC and AUPRC')
        figure.savefig(os.path.join(self.results, f'{self.label_column}_test_results.png'), dpi=300)


# noinspection PyBroadException
class MultiOutputPredictor(SingleOutputPredictor):
    """
    Extension class loads a TF model and classifies images for a given primary and auxiliary label.
    """

    def __init__(self, **kwargs):

        """
        Init for the multi-output predictor.

        *** Kwargs ***

        :keyword aux_column: The name of the auxiliary label column
        :keyword aux_dict: Dictionary of the auxiliary label name and value pairs
        """

        # Add the aux column and aux_dict attributes
        self.aux_column = kwargs['aux_column']
        self.aux_dict = kwargs['aux_dict']
        self.aux_nclasses = len(self.aux_dict.keys())

        super().__init__(**kwargs)

    def _get_df(self):
        """
        Function to load the csv file to a df and select the correct subgroups

        :return: Pandas dataframe
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

        # Reset index after dropping examples
        self.img_df = select_img_df.reset_index(drop=True)

        # Print Metrics
        print('Total images = {}, Images dropped = {}\n'.format(self.img_df.index.size, dropped_images))

    def _make_dataset(self, df):
        """
        Helper function that takes a pandas DF and returns a TF dataset

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

        # Make tf dataset
        tf_ds = tf.data.Dataset.from_tensor_slices((df['Path'], laterality))

        return tf_ds

    def get_predictions(self):
        """
        Main function that returns model predictions for the main and aux labels.
        """

        print('Getting Predictions')

        if self.label_column not in ['Laterality', 'Retinal_Presence']:
            print('Left eye images will be flipped to right eye orientation')

        # Load the model
        model = models.load_model(self.model_path)

        # get the dataset
        pred_ds = self._make_dataset(self.img_df)

        # Map to the image decoder
        pred_ds = pred_ds.map(lambda x, y: self._parse_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        pred_ds = pred_ds.batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Predict
        predictions_primary, predictions_aux = model.predict(pred_ds, verbose=1)

        # If binary return the prediction probability of the +ve class otherwise return the argmax
        if predictions_primary.shape[1] == 1:
            predictions_primary = predictions_primary

        else:
            predictions_primary = predictions_primary.argmax(axis=1)

        if predictions_aux.shape[1] == 1:
            predictions_aux = predictions_aux

        else:
            predictions_aux = predictions_aux.argmax(axis=1)

        # Make the predictions into a dataframe
        predictions_primary_df = pandas.DataFrame(data=[*predictions_primary], columns=[f'{self.label_column}_P'])
        predictions_aux_df = pandas.DataFrame(data=[*predictions_aux], columns=[f'{self.aux_column}_P'])

        # Add the predictions to the path only dataframe by concat
        combined_df = pandas.concat([self.img_df, predictions_primary_df, predictions_aux_df], axis=1)

        # Save predictions
        combined_df.to_csv(os.path.join(self.results, f'{self.label_column}_{self.aux_column}_Predictions_DF.csv'),
                           index=False)

    def test_model(self):
        """
        Main function that tests the model and saves the confusion matrix, ROC and PRC per class for the primary and
        auxiliary tasks.
        """

        # Get the test datasets
        test_ds = self._make_dataset(self.img_df)

        # Map to the image decoder
        test_ds = test_ds.map(lambda x, y: self._parse_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        test_ds = test_ds.batch(batch_size=self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Get labels as one hot encoded values to plot individual outcomes
        primary_labels = tf.one_hot(self.img_df[self.label_column], self.nclasses).numpy()

        aux_labels = tf.one_hot(self.img_df[self.aux_column], self.aux_nclasses).numpy()

        # Load the model
        model = models.load_model(self.model_path)

        # Compute test steps
        test_steps = int(self.img_df.index.size / self.batch_size) + 1

        # Use the model to predict the values from the validation dataset for the label column
        predictions_primary, predictions_aux = model.predict(test_ds, steps=test_steps, verbose=1)

        # Get the primary predictions of both the positive and negative class if binary to enable per class metrics
        if self.nclasses == 2:
            predictions_primary = np.column_stack((1 - predictions_primary, predictions_primary))
        else:
            predictions_primary = predictions_primary

        # Get the aux predictions of both the positive and negative class if binary to enable per class metrics
        if self.aux_nclasses == 2:
            predictions_aux = np.column_stack((1 - predictions_aux, predictions_aux))
        else:
            predictions_aux = predictions_aux

        # Calculate the confusion matrix
        cm_primary = tf.math.confusion_matrix(primary_labels.argmax(axis=1), predictions_primary.argmax(axis=1),
                                              num_classes=self.nclasses).numpy()
        cm_aux = tf.math.confusion_matrix(aux_labels.argmax(axis=1), predictions_aux.argmax(axis=1),
                                          num_classes=self.aux_nclasses).numpy()

        # CM plot
        figure, ax = plt.subplots(nrows=2, ncols=3, figsize=(24, 16))

        cm_plot_primary = ConfusionMatrixDisplay(confusion_matrix=cm_primary, display_labels=self.class_dict.keys())
        cm_plot_primary.plot(ax=ax[0, 0])
        cm_plot_primary.figure_.set_label('Count')
        ax[0, 0].set_title('Confusion Matrix: {}'.format(self.label_column))

        cm_plot_aux = ConfusionMatrixDisplay(confusion_matrix=cm_aux, display_labels=self.aux_dict.keys())
        cm_plot_aux.plot(ax=ax[1, 0])
        cm_plot_aux.figure_.set_label('Count')
        ax[1, 0].set_title('Confusion Matrix: {}'.format(self.aux_column))

        # Plot ROC and PRC per class for the primary label
        for i, c in enumerate(self.class_dict.keys()):
            fpr_primary, tpr_primary, _ = roc_curve(primary_labels[:, i], predictions_primary[:, i])
            class_auc_primary = auc(fpr_primary, tpr_primary)
            ax[0, 1].plot(fpr_primary, tpr_primary, linewidth=2, linestyle='--',
                          label='{0}: {1:.3f}'.format(c, class_auc_primary))

            p_primary, r_primary, _ = precision_recall_curve(primary_labels[:, i], predictions_primary[:, i])
            # Fixes the PR curve at 0 precision then recall is 1 (i.e. if the threshold value is 0).
            p_primary = np.insert(p_primary, 0, 0)
            r_primary = np.insert(r_primary, 0, 1)
            class_pr_auc_primary = auc(r_primary, p_primary)
            ax[0, 2].plot(r_primary, p_primary, linewidth=2, linestyle='--',
                          label='{0}: {1:.3f}'.format(c, class_pr_auc_primary))

        ax[0, 1].legend(loc="lower right", title="AUROC", fancybox=True)
        ax[0, 1].set_title('ROC: {} Classifier'.format(self.label_column))
        ax[0, 1].set_xlabel('1 - Specificity')
        ax[0, 1].set_ylabel('Sensitivity')

        ax[0, 2].legend(loc="lower left", title="AUPRC", fancybox=True)
        ax[0, 2].set_title('PRC: {} Classifier'.format(self.label_column))
        ax[0, 2].set_xlabel('Recall [Sensitivity]')
        ax[0, 2].set_ylabel('Precision')

        # Plot ROC and PRC per class for the aux label
        for i, c in enumerate(self.aux_dict.keys()):
            fpr_aux, tpr_aux, _ = roc_curve(aux_labels[:, i], predictions_aux[:, i])
            class_auc_aux = auc(fpr_aux, tpr_aux)
            ax[1, 1].plot(fpr_aux, tpr_aux, linewidth=2, linestyle='--',
                          label='{0}: {1:.3f}'.format(c, class_auc_aux))

            p_aux, r_aux, _ = precision_recall_curve(aux_labels[:, i], predictions_aux[:, i])
            # Fixes the PR curve at 0 precision then recall is 1 (i.e. if the threshold value is 0).
            p_aux = np.insert(p_aux, 0, 0)
            r_aux = np.insert(r_aux, 0, 1)
            class_pr_auc_aux = auc(r_aux, p_aux)
            ax[1, 2].plot(r_aux, p_aux, linewidth=2, linestyle='--', label='{0}: {1:.3f}'.format(c, class_pr_auc_aux))

        ax[1, 1].legend(loc="lower right", title="AUROC", fancybox=True)
        ax[1, 1].set_title('ROC: {} Classifier'.format(self.aux_column))
        ax[1, 1].set_xlabel('1 - Specificity')
        ax[1, 1].set_ylabel('Sensitivity')

        ax[1, 2].legend(loc="lower left", title="AUPRC", fancybox=True)
        ax[1, 2].set_title('PRC: {} Classifier'.format(self.aux_column))
        ax[1, 2].set_xlabel('Recall [Sensitivity]')
        ax[1, 2].set_ylabel('Precision')

        figure.suptitle('Confusion Matrix, AUROC and AUPRC')
        figure.savefig(os.path.join(self.results, f'{self.label_column}_{self.aux_column}_test_results.png'), dpi=300)
