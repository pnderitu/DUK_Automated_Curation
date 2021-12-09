# Utility to parse arguments
# Paul Nderitu and Diabetes UK 2021

def parse_config(args):
    # Convert args to dct
    config_dict = vars(args)

    # Add the label class dictionaries
    if config_dict['label_column'] == 'Laterality':
        config_dict.update({'class_dict': {'Right': 0, 'Left': 1, 'N/A': 2}})
    elif config_dict['label_column'] == 'Retinal_Status':
        config_dict.update({'class_dict': {'Non-Retinal': 0, 'Retinal': 1}})
    elif config_dict['label_column'] == 'Retinal_Field':
        config_dict.update({'class_dict': {'Macula': 0, 'Nasal': 1, 'ORF': 2}})
    elif config_dict['label_column'] == 'Gradability':
        config_dict.update({'class_dict': {'Ungradable': 0, 'Gradable': 1}})
    else:
        config_dict.update({'class_dict': None})

    # Add the label class dictionaries
    if config_dict['aux_column'] == 'Laterality':
        config_dict.update({'aux_dict': {'Right': 0, 'Left': 1, 'N/A': 2}})
    elif config_dict['aux_column'] == 'Retinal_Status':
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
        'image_size': [224, 224],
        'patience': 3,
        'max_epochs': 50,
        'hparams': {
            "dropout": config_dict['dropout'],
            "learning_rate": config_dict['learning_rate'],
            "regularisation": config_dict['regularisation']}
    }

    # Keeps all default values not overwritten by the passed config
    default_dict.update(config_dict)

    return default_dict
