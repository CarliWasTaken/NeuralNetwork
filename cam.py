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
#TODO: Purify Lines
#TODO: compress image
def process_image(image):
    h, w, c = image.shape

    crop_img = image[int((h/3)*2):h, 0:w]

    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_img,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(crop_img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(crop_img, 0.8, line_image, 1, 0)
    image = draw_lines(lines_edges)

    return image

def draw_lines(img):
    h, w, c = img.shape

    for i in range(3):
        start_point = (0, int(((h-20)/2)*i)+10)
        end_point = (w, int(((h-20)/2)*i)+10)
        color = (0, 0, 0)
        thickness = 3

        img = cv2.line(img, start_point, end_point, color, thickness)

    return img

if __name__ == "__main__":
    get_data_point()