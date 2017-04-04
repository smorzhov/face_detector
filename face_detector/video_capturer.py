import threading
import Queue
import cv2


class VideoCapturer:
    """It recognize faces in video"""

    def __init__(self, webcam_id):
        if webcam_id + 1 > -1:
            self.webcam_id = webcam_id
        else:
            raise ValueError("webcam_id must be a positive integer")
        self.frame_captured = threading.Event()
        self.frame_captured.clear()
        self.frame_processed = threading.Event()
        self.frame_processed.clear()
        self.raw_frames = Queue.Queue()
        self.processed_frames = Queue.Queue()
        self.reading_stopped = threading.Event()
        self.reading_stopped.clear()

    def capture(self, process_func):
        """It captures video from a web camera"""
        processing = threading.Thread(
            target=self._process_frames, args=(process_func,))
        processing.start()
        showing = threading.Thread(target=self._show_frames)
        showing.start()
        # waiting until all threads finish their work
        self._read_frames()
        processing.join()
        showing.join()

    @staticmethod
    def release_capture(capture):
        """It releases the capture"""
        capture.release()
        cv2.destroyAllWindows()

    def _read_frames(self):
        """It reads frames and puts them into the queue"""
        capture = cv2.VideoCapture(self.webcam_id)
        while not self.reading_stopped.isSet():
            ret, frame = capture.read()
            if ret:
                self.raw_frames.put(frame)
                if not self.frame_captured.isSet():
                    self.frame_captured.set()
            else:
                VideoCapturer.release_capture(capture)
                return
        capture.release()

    def _process_frames(self, process_func):
        """It processes frames"""
        self.frame_captured.wait()
        while True:
            try:
                item = self.raw_frames.get(True, 1)
            except Queue.Empty:
                self.frame_captured.clear()
                break
            else:
                self.processed_frames.put(process_func(item))
                self.raw_frames.task_done()
                if not self.frame_processed.isSet():
                    self.frame_processed.set()

    def _show_frames(self):
        """It shows frames"""
        self.frame_processed.wait()
        while True:
            try:
                item = self.processed_frames.get(True, 1)
            except Queue.Empty:
                self.frame_processed.clear()
                break
            else:
                cv2.imshow('frame', item)
                self.processed_frames.task_done()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.reading_stopped.set()
        cv2.destroyAllWindows()
