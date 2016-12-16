from pathlib import Path
import numpy as np
import cv2
import sys
import os
from video_capturer import VideoCapturer

RESIZE_FACTOR = 4


class RecognizeFisherFaces:
    """It recognizes faces on video"""

    def __init__(self):
        """RecognizeFisherFaces constructor"""
        casc_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(casc_path)
        self.face_dir = 'face_data'
        self.model = cv2.face.createFisherFaceRecognizer()
        self.face_names = []

    def recognize(self, webcam_id):
        """It recofnizes peoples' faces on video"""
        try:
            self._load_trained_data()
            capturer = VideoCapturer(webcam_id)
            capturer.capture(self._process_image)
        except ValueError as err:
            print("Error occure: {0}".format(err))
        except:
            print("Unexpected error: {0}".format(sys.exc_info()[0]))
            raise

    def _load_trained_data(self):
        """It loads trained data"""
        names = {}
        key = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names
        trained_data = 'fisher_trained_data.xml'
        if Path(trained_data).is_file():
            self.model.load(trained_data)

    def _process_image(self, input_img):
        """It recognizes faces on the image"""
        frame = cv2.flip(input_img, 1)
        resized_width, resized_height = (112, 92)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(
            gray, (int(gray.shape[1] / RESIZE_FACTOR), int(gray.shape[0] / RESIZE_FACTOR)))
        faces = self.face_cascade.detectMultiScale(
            gray_resized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        persons = []
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            label, confidence = self.model.predict(face_resized)
            if confidence < 300:
                person = self.names[label]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(
                    frame,
                    '%s - %.0f' % (person, confidence),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                person = 'Unknown'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, person, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            persons.append(person)
        return frame
        #return (frame, persons)
