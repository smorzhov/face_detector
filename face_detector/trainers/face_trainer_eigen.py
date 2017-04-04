import cv2
from face_detector.trainers.face_trainer import FaceTrainer


class EigenFaceTrainer(FaceTrainer):
    """It trains classifier to detect a person's face with Eigen algorithm"""

    def __init__(self, face_name, face_dir, casc_path):
        """TrainFisherFaces constructor"""
        super(EigenFaceTrainer, self).__init__(face_name, face_dir, casc_path)
        self.model = cv2.createEigenFaceRecognizer()


    def capture_training_images(self, webcam_id):
        """It captures video from a webcame with a given id"""
        super(EigenFaceTrainer, self).capture_training_images(webcam_id)

    def process_image(self, input_img):
        """It detects face on the image"""
        return super(EigenFaceTrainer, self).process_image(input_img)

    def train_data(self, training_data_path):
        """It trains classifier"""
        super(EigenFaceTrainer, self).train_data(training_data_path)
