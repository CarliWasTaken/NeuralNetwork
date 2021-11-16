import cv2

class Camera:
    def take_picture(self, dev:bool) -> list[int]:
        if(dev):
            return cv2.imread("testImages/picture.png")
        else:
            cam = cv2.VideoCapture(0)
            ret, image = cam.read()
            return image
    pass
