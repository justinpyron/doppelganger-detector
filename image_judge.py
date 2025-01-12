import cv2 as cv
import numpy as np

class ImageJudge:
    def __init__(self):
        path = "haarcascade_frontalface_default.xml"
        self.face_detector = cv.CascadeClassifier(path)

    def extract_faces(self, image: np.array) -> np.array:
        """Extracts bounding boxes around faces in an image"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face_rectangles = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5, 
            minSize=(10, 10),
            flags = cv.CASCADE_SCALE_IMAGE,
        )
        return face_rectangles
    
    def image_is_acceptable(
        self,
        image: np.array,
        min_face_fraction: float,
        verbose: bool = False,
    ) -> bool:
        """
        Determine if an image meets two criteria:
        (1) Contains only a single face
        (2) The face occupies a minimum fraction of the image's pixels
        """
        face_rectangles = self.extract_faces(image)
        if len(face_rectangles) != 1:
            if verbose:
                print("The photo does not contain (only) one person!")
            return False
        x, y, width, height = face_rectangles[0]
        pixels_image = image.shape[0] * image.shape[1]
        pixels_face = width * height
        ratio = pixels_face / pixels_image
        if ratio < min_face_fraction:
            if verbose:
                print(f"Face too small! (ratio = {ratio:.3f})")
            return False
        else:        
            return True
