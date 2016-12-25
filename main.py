import argparse
import json
import io
from face_trainer_eigen import EigenFaceTrainer
from face_trainer_fisher import FisherFaceTrainer
from face_recognizer_eigen import EigenFacesRecognizer
from face_recognizer_fisher import FisherFacesRecognizer


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="It recognize people faces on the video")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", nargs='?', help="store traininig data")
    group.add_argument("-T", "--training", action="store_true", help="train classifier")
    group.add_argument("-r", "--recognize",
                       action="store_true", help="recognize people")
    types = parser.add_mutually_exclusive_group()
    types.add_argument("-f", "--fisher", action="store_true",
                       help="Fisher face algorithm")
    types.add_argument("-e", "--eigen", action="store_true",
                       help="Eigen face algorithm")
    parser.add_argument("-c", "--camera-id", nargs='?',
                        help="camera id", default=0, type=int)
    args = parser.parse_args()
    try:
        config = load_config('./config.json')
    except ValueError as error:
        print(error)
        return
    if args.fisher:
        algo_type = 'fisher'
    else:
        algo_type = 'eigen'
    if args.train:
        trainer = assign_trainer(algo_type, args.train, config)
        trainer.capture_training_images(
            args.camera_id)
        return
    if args.training:
        trainer = assign_trainer(algo_type, args.train, config)
        trainer.train_data(config[algo_type]['path'] + config[algo_type]['training_data'])
        return
    if args.recognize:
        recognizer = assign_recognizer(algo_type, config)
        recognizer.recognize(
            args.camera_id,
            config[algo_type]['path'] + config[algo_type]['training_data'])
        return


def assign_trainer(trainer_type, user, config):
    """It assigns trainer"""
    if trainer_type is 'fisher':
        return FisherFaceTrainer(
            user,
            config[trainer_type]['path'] + config[trainer_type]['face_data'],
            config['cascade_path'])
    elif trainer_type is 'eigen':
        return EigenFaceTrainer(
            user,
            config[trainer_type]['path'] + config[trainer_type]['face_data'],
            config['cascade_path'])
    else:
        return None


def assign_recognizer(recognizer_type, config):
    """It assignes recognizer"""
    if recognizer_type is 'fisher':
        return FisherFacesRecognizer(
            config['fisher']['path'] + config['fisher']['face_data'],
            config['cascade_path'])
    elif recognizer_type is 'eigen':
        return EigenFacesRecognizer(
            config['eigen']['path'] + config['eigen']['face_data'],
            config['cascade_path'])
    else:
        return None


def load_config(file_name):
    """It loads configuration file"""
    with io.open(file_name, 'r', encoding='utf8') as config_file:
        config = json.load(config_file)
        if (config['eigen'] is None or
                config['fisher'] is None or
                not isinstance(config['cascade_path'], basestring)):
            raise ValueError("config.json has an unappropriate data")
        if check_config(config['eigen']) and check_config(config['fisher']):
            return config
        else:
            raise ValueError("config.json has an unappropriate data")


def check_config(data):
    """Checks whether the given json has an appropriate data"""
    return (isinstance(data['path'], basestring) and
            isinstance(data['face_data'], basestring) and
            isinstance(data['training_data'], basestring))


if __name__ == "__main__":
    main()
