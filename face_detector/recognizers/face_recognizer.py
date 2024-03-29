from abc import ABCMeta, abstractmethod
import sys
import os
import cv2
from face_detector.video_capturer import VideoCapturer

class FaceRecognizer:
    """Base class for face recognizer"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, face_dir, casc_path):
        """RecognizeFisherFaces constructor"""
        self.face_cascade = cv2.CascadeClassifier(casc_path)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.face_dir = face_dir
        self.face_names = []
        self.model = None

    @abstractmethod
    def recognize(self, webcam_id, training_data_path):
        """It recofnizes people's faces on video"""
        try:
            self.load_trained_data(training_data_path)
            capturer = VideoCapturer(webcam_id)
            capturer.capture(self.process_image)
        except ValueError as err:
            print("Error occure: {0}".format(err))
        except:
            print("Unexpected error: {0}".format(sys.exc_info()[0]))
            raise

    @abstractmethod
    def load_trained_data(self, training_data):
        """It loads trained data"""
        names = {}
        key = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names
        self.model.load(training_data)

    @abstractmethod
    def process_image(self, input_img):
        """It recognizes faces on the image"""
        pass

    def normalize_image(self, image):
        """It implements an adaptive histogram equalization to the image"""
        return self.clahe.apply(image)
