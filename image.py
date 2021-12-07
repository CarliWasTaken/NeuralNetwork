import cv2
import numpy as np
from editor import ImageEditor
from cam import Camera

class Image:
    HSV_LOWER = [40, 0, 80]
    HSV_UPPER = [180, 150, 225]

    def __init__(self, dev: bool):
        self.__image: list[int] = []
        self.__camera: Camera = Camera()
        self.__editor: ImageEditor = ImageEditor()
        self.__dev = dev

        # Get steering angle and speed
        self.__angle: int = 0
        self.__speed: int = 0
        pass

    def set_HSV_LOWER(self, lower):
        if len(lower) != 3:
            print("rejected")
            return
        self.HSV_LOWER = lower
        pass

    def set_HSV_UPPER(self, upper):
        if len(upper) != 3:
            print("rejected")
            return
        self.HSV_LOWER = upper
        pass


    def get_data_point(self) -> np.array:
        self.__image = self.__camera.take_picture(self.__dev)

        # Process image
        self.__process_image_Mask()

        self.__image = self.__editor.resize_image(self.__image)

        arr: list[int] = np.zeros((60,60))
        for point in self.__detect_intersections():
            arr[point[0]][point[1]] = 1
            pass
        
        # self.__image = self.__editor.draw_lines(self.__image, self.__get_line_heights)
        # cv2.imshow("1", self.__image)
        # cv2.waitKey(0)

        # Put frame, speed and angle together
        data_point: list = np.array([arr, self.__speed, self.__angle], dtype=object)

        return data_point

    def set_speed_and_angle(self, speed: int, angle: int):
        self.__speed = speed
        self.__angle = angle

    def get_speed_and_angle(self) -> 'tuple[int, int]':
        return (self.__speed, self.__angle)

    
    def __get_line_heights(self):
        h, w = self.__image.shape
        for i in range(3):
            yield int((h/4)*(i+1))
            pass
        pass

    def __detect_intersections(self) -> 'list[int]':
        points = []

        for height in self.__get_line_heights():
            index = 0
            while index < len(self.__image[height])//2:
                if self.__image[height][index] == 255:
                    points.append((height, index))
                    break
                index += 1
                pass

            index = len(self.__image[height])-1
            while index >= len(self.__image[height])//2:
                if self.__image[height][index] == 255:
                    points.append((height, index))
                    break
                index -=1
                pass
        return points
        pass

    def __process_image_Mask(self):
        self.__image = self.__editor.crop_image(self.__image)
        
        gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2HSV)
        blur = cv2.medianBlur(gray, 5)
        lower: list = np.array(self.HSV_LOWER, dtype="uint8")
        upper: list = np.array(self.HSV_UPPER, dtype="uint8")
        mask = cv2.inRange(blur, lower, upper)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.fillPoly(mask, cnts, (255,255,255))

        self.__image = mask
        pass