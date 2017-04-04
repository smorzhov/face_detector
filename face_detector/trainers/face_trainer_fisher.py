import os
import cv2
from face_detector.video_capturer import VideoCapturer
from face_detector.trainers.face_trainer import FaceTrainer


FREQ_DIV = 5  # frequency divider for capturing training images
RESIZE_FACTOR = 4
NUM_TRAINING = 100


class FisherFaceTrainer(FaceTrainer):
    """It trains classifier to detect a person's face with Fisher algorithm"""

    def __init__(self, face_name, face_dir, casc_path):
        super(FisherFaceTrainer, self).__init__(face_name, face_dir, casc_path)
        self.model = cv2.createFisherFaceRecognizer()

    def capture_training_images(self, webcam_id):
        """It captures video from a webcame with a given id"""
        try:
            capturer = VideoCapturer(webcam_id)
            capturer.capture(self.process_image)
        except ValueError as err:
            print("Error occure: {0}".format(err))

    def process_image(self, input_img):
        """It detects face on the image"""
        return super(FisherFaceTrainer, self).process_image(input_img)

    def are_enough_faces(self):
        """Were there enough faces captured"""
        existing_faces = 0
        for (_, dirs) in os.walk(self.face_dir):
            for _ in dirs:
                existing_faces += 1

        return existing_faces > 1

    def train_data(self, training_data_path):
        """It trains classifier"""
        if self.are_enough_faces():
            super(FisherFaceTrainer, self).train_data(training_data_path)
        else:
            print("One more user needed to start recognition!")
