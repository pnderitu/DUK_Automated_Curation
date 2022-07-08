#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022, Paul Nderitu
# All rights reserved.
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# Utility to parse arguments

def parse_config(args):
    # Convert args to dct
    config_dict = vars(args)

    # Add the label class dictionaries
    if config_dict['label_column'] == 'Laterality':
        config_dict.update({'class_dict': {'Right': 0, 'Left': 1, 'Unidentifiable': 2}})
    elif config_dict['label_column'] == 'Retinal_Presence':
        config_dict.update({'class_dict': {'Non-Retinal': 0, 'Retinal': 1}})
    elif config_dict['label_column'] == 'Retinal_Field':
        config_dict.update({'class_dict': {'Macula': 0, 'Nasal': 1, 'ORF': 2}})
    elif config_dict['label_column'] == 'Gradability':
        config_dict.update({'class_dict': {'Ungradable': 0, 'Gradable': 1}})
    else:
        config_dict.update({'class_dict': None})

    # Add the label class dictionaries
    if config_dict['aux_column'] == 'Laterality':
        config_dict.update({'aux_dict': {'Right': 0, 'Left': 1, 'Unidentifiable': 2}})
    elif config_dict['aux_column'] == 'Retinal_Presence':
        config_dict.update({'aux_dict': {'Non-Retinal': 0, 'Retinal': 1}})
    elif config_dict['aux_column'] == 'Retinal_Field':
        config_dict.update({'aux_dict': {'Macula': 0, 'Nasal': 1, 'ORF': 2}})
    elif config_dict['aux_column'] == 'Gradability':
        config_dict.update({'aux_dict': {'Ungradable': 0, 'Gradable': 1}})
    else:
        config_dict.update({'aux_dict': None})

    # Add default values
    default_dict = {
        'save_examples': True,
        'seed': 7,
        'image_size': [config_dict['image_height'], config_dict['image_width']],
        'patience': 3,
        'max_epochs': config_dict['max_epochs'],
        'hparams': {
            "dropout": config_dict['dropout'],
            "learning_rate": config_dict['learning_rate'],
            "regularisation": config_dict['regularisation']}
    }

    # Keeps all default values not overwritten by the passed config
    default_dict.update(config_dict)

    return default_dict
