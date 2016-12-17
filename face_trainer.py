import numpy as np
import cv2
import sys
import os
from video_capturer import VideoCapturer

FREQ_DIV = 5  # frequency divider for capturing training images
RESIZE_FACTOR = 4
NUM_TRAINING = 100


class TrainEigenFaces:
    """It trains classifier to detect a person's face"""

    def __init__(self, face_name):
        """TrainFisherFaces constructor"""
        casc_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(casc_path)
        self.face_dir = 'face_data'
        self.face_name = face_name
        self.path = os.path.join(self.face_dir, self.face_name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.model = cv2.createEigenFaceRecognizer()
        self.count_captures = 0
        self.count_timer = -1

    def capture_training_images(self, webcam_id):
        """It captures video from a webcame with a given id"""
        try:
            capturer = VideoCapturer(webcam_id)
            capturer.capture(self._process_image)
            self._train_data()
        except ValueError as err:
            print("Error occure: {0}".format(err))

    def _process_image(self, input_img):
        """It detects face on the image"""
        frame = cv2.flip(input_img, 1)
        resized_width, resized_height = (112, 92)
        if self.count_captures < NUM_TRAINING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(
                gray, (int(gray.shape[1] / RESIZE_FACTOR), int(gray.shape[0] / RESIZE_FACTOR)))
            faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            if len(faces) > 0:
                self.count_timer += 1
                areas = []
                for (x, y, w, h) in faces:
                    areas.append(w * h)
                max_area, idx = max([(val, idx)
                                     for idx, val in enumerate(areas)])
                face_sel = faces[idx]

                x = face_sel[0] * RESIZE_FACTOR
                y = face_sel[1] * RESIZE_FACTOR
                w = face_sel[2] * RESIZE_FACTOR
                h = face_sel[3] * RESIZE_FACTOR

                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(
                    face, (resized_width, resized_height))
                img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(
                    self.path) if fn[0] != '.'] + [0])[-1] + 1

                if self.count_timer % FREQ_DIV == 0:
                    self.count_timer = -1
                    cv2.imwrite('%s/%s.png' %
                                (self.path, img_no), face_resized)
                    self.count_captures += 1
                    print("Captured image: {0}".format(self.count_captures))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, self.face_name, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        elif self.count_captures == NUM_TRAINING:
            print("Enough data was captured. Press 'q' to continue ...")
            self.count_captures += 1

        return frame

    def _train_data(self):
        """It trains classifier"""
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]
        print("Training ...")
        self.model.train(imgs, tags)
        print("Saving result ...")
        self.model.save('eigen_trained_data.xml')
        print("Completed successfully!")
        return
