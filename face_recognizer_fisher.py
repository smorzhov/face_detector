import cv2
from face_recognizer import FaceRecognizer

RESIZE_FACTOR = 4


class FisherFacesRecognizer(FaceRecognizer):
    """It recognizes faces on video using Fisher face recognition algorithm"""

    def __init__(self, face_dir, casc_path):
        super(FisherFacesRecognizer, self).__init__(face_dir, casc_path)
        self.model = cv2.createFisherFaceRecognizer()

    def recognize(self, webcam_id, training_data_path):
        """It recofnizes people's faces on video"""
        super(FisherFacesRecognizer, self).recognize(
            webcam_id, training_data_path)

    def load_trained_data(self, training_data):
        """It loads trained data"""
        super(FisherFacesRecognizer, self).load_trained_data(training_data)

    def process_image(self, input_img):
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
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
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
            confidence = self.model.predict(self.normalize_image(face_resized))
            if confidence[1] < 130:
                person = self.names[confidence[0]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(
                    frame,
                    '%s - %.0f' % (person, confidence[1]),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 255, 0))
            else:
                person = 'Unknown'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, person, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            persons.append(person)
        return frame
