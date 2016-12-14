from face_detector import FaceDetector


def main():
    """Main function"""
    try:
        detector = FaceDetector(0)
        detector.capture()
    except ValueError as err:
        print("Error occure: " + err)


if __name__ == "__main__":
    main()
