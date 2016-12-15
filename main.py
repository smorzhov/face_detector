import argparse
from face_trainer import TrainFisherFaces


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="It recognize people faces on the video")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", nargs='?', help="train classifier")
    group.add_argument("-r", "--recognize", help="recognize people")
    parser.add_argument("-c", "--camera-id", nargs='?', help="camera id", default=0, type=int)
    args = parser.parse_args()
    if args.train:
        trainer = TrainFisherFaces(args.train)
        trainer.capture_training_images(args.camera_id)
        return
    if args.recognize:
        return


if __name__ == "__main__":
    main()
