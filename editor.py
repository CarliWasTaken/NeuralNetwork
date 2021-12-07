import cv2

class ImageEditor:
    def resize_image(self, image: 'list[int]'):
        dim = (60, 60)
        return cv2.resize(image, dim)
        pass


    def crop_image(self, image: 'list[int]'):
        h, w, c = image.shape
        return image[int((h/3)*2):h, 0:w]
        pass

    def draw_lines(self, image: 'list[int]', get_line_heights):
        h, w = image.shape

        for height in get_line_heights():
            start_point = (0, height)
            end_point = (w, height)
            color = (255, 255, 255)
            thickness: int = 1

            image = cv2.line(image, start_point, end_point, color, thickness)
        pass
        return image
    pass