import cv2
import numpy as np
import glob
import os.path

def create_dataset(img_url):
    training_img = []
    training_label = []

    for img in glob.glob(img_url):
        image = cv2.imread(img)
        gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)

        _, contours, _ = cv2.findContours(thresh.copy(),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for cnt in contours:
            rectangles.append(cnt)

        numbers = []
        for num in rectangles:
            x, y, w, h = cv2.boundingRect(num)
            number = np.zeros((h,w), np.uint8)
            number[:h, :w] = thresh[y:y + h, x:x + w]
            number = img_resize(number, 28, 28)
            flat = number.reshape(-1, 28*28).astype(np.float32)
            training_img.append(flat)

            training_label.append(int(os.path.basename(img)[0]))

    training_label = np.array(training_label).reshape(len(training_label), 1)
    training_img = np.array(training_img).reshape(len(training_img), -1)

    return training_img, training_label

def img_resize(image, height, width):
    shape = (height,width)
    h, w = image.shape
    w = width if w > width else w
    h = height if h > height else h

    x_offset = abs(width - w)
    y_offset = abs(height - h)

    frame = np.zeros((height,width), np.uint8)

    x_start = x_offset / 2 if x_offset != 0 else 0
    y_start = y_offset / 2 if y_offset != 0 else 0

    frame[y_start:h + y_start, x_start:w + x_start] = image[0:h, 0:w]

    return frame
