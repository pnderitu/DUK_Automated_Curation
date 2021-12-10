# Automated Image Curation Main
# Paul Nderitu & Diabetes UK 2021

import argparse
from utils import parse_config
from tf_trainers import SingleOutputClassifier, MultiOutputClassifier
from tf_predict import SingleOutputPredictor, MultiOutputPredictor

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--df_path', type=str,
                        help='Path with the csv file containing the Image_IDs, patient ID [PID] (optional) +/- labels')
    parser.add_argument('-ip', '--image_path', type=str,
                        help='Path to the images')
    parser.add_argument('-sp', '--save_path', type=str,
                        help='Path to save logs, models, results and examples')

    parser.add_argument('-l', '--label_column', type=str, default='Laterality',
                        help="Name of the label column (default: Laterality)")

    parser.add_argument('-al', '--aux_column', type=str, default='Retinal_Presence',
                        help="Name of the auxiliary label column if training or predicting using a multi-output model"
                             "(default: Retinal_Presence)")

    parser.add_argument('-mt', '--model_type', type=str, default='multi-output',
                        help='Select the modelling approach (default: multi-output)')

    parser.add_argument('-m', '--mode', type=str, default="train",
                        help='Select the mode. Must be  tune, train, test or predict (default: train)')

    parser.add_argument('-ucw', '--use_class_weights', type=bool, default=True,
                        help="If class weights should be used (single-output models only) (default: True)")
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help="Batch size to use (default: 32)")
    parser.add_argument('-do', '--dropout', type=float, default=0.2,
                        help="Dropout (default: 0.2)")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument('-r', '--regularisation', type=float, default=0.001,
                        help="Regularisation (default: 0.001)")

    parser.add_argument('-mp', '--model_path', type=str,
                        help='Path to a saved Tensorflow model if testing or predicting')

    # Get args and add config
    args = parser.parse_args()
    config = parse_config(args)

    # Some assertions
    assert config['df_path'] is not None, 'No path to the csv file of images +/- labels provided'
    assert config['image_path'] is not None, 'No path to images provided'
    assert config['save_path'] is not None, 'No save path provided'

    # Model type
    if config['model_type'] == 'single-output' and config['mode'] in ['train', 'tune']:
        print('-------------- Selected model_type: single-output --------------\n')
        trainer = SingleOutputClassifier(**config)

    elif config['model_type'] == 'single-output' and config['mode'] in ['test', 'predict']:
        print('-------------- Selected model_type: single-output --------------\n')
        trainer = SingleOutputPredictor(**config)

    elif config['model_type'] == 'multi-output' and config['mode'] in ['train', 'tune']:
        print('-------------- Selected model_type: multi-output --------------\n')
        trainer = MultiOutputClassifier(**config)

    elif config['model_type'] == 'multi-output' and config['mode'] in ['test', 'predict']:
        print('-------------- Selected model_type: multi-output --------------\n')
        trainer = MultiOutputPredictor(**config)

    else:
        raise AttributeError("Invalid model_type and mode provided, must be 'single-output' or 'multi-output' "
                             "model_type with either 'train', 'tune', 'test' or 'predict' modes")

    # Run training, tuning, testing or prediction as per the mode
    # Tune
    if config['mode'] == 'tune':
        assert config['label_column'] is not None or config['aux_column'], 'No labels provided'
        print('-------------- Selected mode: tune --------------\n')
        trainer.tune_model()

    # Train
    if config['mode'] == 'train':
        assert config['label_column'] is not None or config['aux_column'], 'No labels provided'
        print('-------------- Selected mode: train --------------\n')
        trainer.train_val_model()

    # Test
    if config['mode'] == 'test':
        assert config['model_path'] is not None, 'No model path provided'
        assert config['label_column'] is not None or config['aux_column'], 'No labels provided'
        print('-------------- Selected mode: test --------------\n')
        trainer.test_model()

    # Predict
    if config['mode'] == 'predict':
        assert config['model_path'] is not None, 'No model path provided'
        assert config['label_column'] is not None or config['aux_column'], 'No labels provided'
        print('-------------- Selected mode: predict --------------\n')
        trainer.get_predictions()
