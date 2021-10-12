import cv2
import numpy as np

def get_data_point():
    # Get image from camera
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    # Process image
    image = process_image(frame)

    # Get steering angle and speed
    angle = 0
    speed = 0

    # Put frame, speed and angle together
    data_point = np.array([image, speed, angle])

    return data_point



# Implementierts de methode
def process_image(image):
    pass

if __name__ == "__main__":
    get_data_point()