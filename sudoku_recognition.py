import cv2
import numpy as np
import glob
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sudoku_solver import sudoku
from knn_classifier import create_dataset, train_knn, img_resize, train_knn, predict

def import_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray.copy(), 5)
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

    return oppening, bounding_rect

def find_numbers(oppening, thresh_img, rectangles_img):
    _, num_contours, _ = cv2.findContours(oppening.astype("uint8").copy(),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    num_rectangle = []
    bounding_rectangles = []
    for num_contour in num_contours:
        x, y, w, h = cv2.boundingRect(num_contour)
        if w > 7 and h > 10 and h < 42:
            bounding_rectangles.append((x, y, w, h))
            num_rectangle.append(num_contour)

    numbers = []
    for number in num_rectangle:
        x, y, w, h = cv2.boundingRect(number)
        cv2.rectangle(rectangles_img, (x - 4,y - 4), (x + w + 3, y + h + 4), (0, 255, 0), 2)
        number = np.zeros((h,w), np.uint8)
        number[:h, :w] = thresh_img[y:y + h, x:x + w]
        number = img_resize(number, 28, 28)
        numbers.append(number)

    ret = [(numbers[i], bounding_rectangles[i]) for i in range(len(numbers))]

    return ret

def intersection_area(boxA, boxB):
    left = max(boxA[0], boxB[0])
    top = max(boxA[1], boxB[1])
    right = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    bottom = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if left <= right and top <= bottom:
        return abs((right - left) * (bottom - top))

    return -1

def import_test(img_url):
    img, thresh = import_image(img_url)
    crop, rectangle = crop_image(thresh)

    x, y, w, h = rectangle
    orig_cropped = np.zeros((h, w, 3), np.uint8)
    orig_cropped[:h, :w, :] = img[y:y+h, x:x+w, :]

    return orig_cropped, crop, rectangle

def filter_test_img(croped_img):
    gray_test = 255 - cv2.cvtColor(croped_img, cv2.COLOR_BGR2GRAY)
    gray_test_origin = gray_test.copy()

    blur_test = cv2.GaussianBlur(gray_test,(5,5),0)
    thresh_test = cv2.adaptiveThreshold(blur_test, 255, 1, 1, 11, 2)

    return thresh_test

def find_contours_test(thresh):
    _, contours, _ = cv2.findContours(thresh.copy(),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for cnt in contours:
        num_test = cnt.copy()
        cx, cy, cw, ch = cv2.boundingRect(num_test)
        if cw > 30 and cw < 70 and ch > 30 and ch < 60:
            areas.append((cx, cy, cw, ch))

    return areas

def sort_numbers(numbers):
    sort_y = sorted(numbers, key=lambda r: r[1])

    rows = []
    for i in range(9):
        sliced_row = sort_y[i * 9:(i + 1) * 9]
        sorted_y = sorted(sliced_row, key=lambda r: r[0])
        rows.append(sorted_y)
    return rows

def set_intersections_postion(numbers, rows):
    intersections = {}

    for i in range(9):
        for j in range(9):
            for idx, (number_image, num_rectangle) in enumerate(numbers):
                area = intersection_area(num_rectangle, rows[i][j])

                if not idx in intersections:
                    intersections[idx] = 0

                if area > 0 and area > intersections[idx]:
                    intersections[idx] = (i, j)

    return intersections

def predict_test_number(numbers, intersections, model):
    sudoku_table = np.zeros((9, 9), np.uint8)

    for number_index, (row, column) in intersections.iteritems():
        recognized_number = predict(numbers[number_index][0], model)
        sudoku_table[row, column] = recognized_number

    return sudoku_table

def main():
    orig_cropped, crop, rectangle = import_test('sudoku_images/image7.jpg')
    find_crop = orig_cropped.copy()
    thresh_test = filter_test_img(find_crop)
    areas = find_contours_test(thresh_test)
    rows = sort_numbers(areas)

    filt_img, rect = filter_image(crop, rectangle)
    numbers = find_numbers(filt_img, crop, orig_cropped)

    samples, labels = create_dataset('dataset/*.jpg')
    model = train_knn(samples, labels)

    intersections = set_intersections_postion(numbers, rows)

    sudoku_table = predict_test_number(numbers, intersections, model)

    success, steps = sudoku(sudoku_table)
    print "Solved: {0} in {1} steps\n".format(success, steps)
    print "SuDoKu Solver:\n", np.matrix(sudoku_table)

    cv2.imshow('Number', orig_cropped)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
