import cv2
import numpy as np
from knn_dataset import img_resize

def import_image(image_path):
    if type(image_path) is str:
        img = cv2.imread(image_path)
    else:
        img = image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray.copy(),(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    return img, thresh

def crop_image(thresh):
    _, contours, _ = cv2.findContours(thresh.copy(),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    mask = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            mask = cnt

    bounding_rect = cv2.boundingRect(mask)
    x, y, w, h = bounding_rect
    oppening = np.zeros((h, w), np.float)
    oppening[0:h, 0:w] = thresh[y:y + h, x:x + w]

    return oppening, bounding_rect

def filter_image(oppening, bounding_rect):
    kernel = np.zeros((11,11),dtype=np.uint8)
    kernel[5,...] = 1
    line = cv2.morphologyEx(oppening,cv2.MORPH_OPEN, kernel, iterations=2)
    oppening-=line

    x, y, w, h = bounding_rect
    oppening_cann = np.empty((h, w), np.uint8)
    oppening_cann[:, :] = oppening[:, :]

    lines = cv2.HoughLinesP(oppening_cann, 1, np.pi/180, 106, 80, 10)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(oppening,(x1,y1),(x2,y2),(0,0,0), 3)

    oppening = cv2.morphologyEx(oppening,cv2.MORPH_OPEN, (2, 3), iterations=1)

    return oppening, bounding_rect

def find_numbers(oppening, rectangles_img):
    _, num_contours, _ = cv2.findContours(oppening.astype("uint8").copy(),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    num_rectangle = []
    bounding_rectangles = []
    for num_contour in num_contours:
        x, y, w, h = cv2.boundingRect(num_contour)
        if w > 5 and h > 10 and h < 42:
            bounding_rectangles.append((x, y, w, h))
            num_rectangle.append(num_contour)

    numbers = []
    for number in num_rectangle:
        x, y, w, h = cv2.boundingRect(number)
        cv2.rectangle(rectangles_img, (x - 4,y - 4), (x + w + 3, y + h + 4), (0, 255, 0), 2)
        number = np.zeros((h,w), np.uint8)
        number[:h, :w] = oppening[y:y + h, x:x + w]
        number = img_resize(number, 28, 28)
        numbers.append(number)

    cv2.imshow('Find_numbers', rectangles_img)
    cv2.waitKey()
    ret = [(numbers[i], bounding_rectangles[i]) for i in range(len(numbers))]

    return ret
