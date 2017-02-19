import cv2
import numpy as np
from sudoku_solver import sudoku
from knn_classifier import KNNClassifier
from main_sudoku_functions import import_image, crop_image, filter_image, find_numbers
import time

def import_test(img_url):
    img, thresh = import_image(img_url)
    crop, rectangle = crop_image(thresh)

    x, y, w, h = rectangle
    orig_cropped = np.zeros((h, w, 3), np.uint8)
    orig_cropped[:h, :w, :] = img[y:y+h, x:x+w, :]

    return orig_cropped, crop, rectangle

def filter_test_img(croped_img):
    gray_test = 255 - cv2.cvtColor(croped_img, cv2.COLOR_BGR2GRAY)

    blur_test = cv2.GaussianBlur(gray_test.copy(),(5,5),0)
    thresh_test = cv2.adaptiveThreshold(blur_test, 255, 1, 1, 11, 2)

    return thresh_test

def find_contours_test(thresh, crop_img):
    thresh = cv2.dilate(thresh, (3, 3), iterations=1)
    _, contours, _ = cv2.findContours(thresh.copy(),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    union_cnt = []
    for cnt in contours:
        num_test = cnt.copy()
        cx, cy, cw, ch = cv2.boundingRect(num_test)
        if cw > 30 and cw < 70 and ch > 30 and ch < 60:
            areas.append((cx, cy, cw, ch))
            union_cnt.append(cnt)

    for first in range(len(union_cnt) - 1):
        first_contour = cv2.boundingRect(union_cnt[first])
        for second in range(first + 1, len(union_cnt)):
            second_contour = cv2.boundingRect(union_cnt[second])

            if distance(first_contour, second_contour) < 8:
                areas.remove(first_contour)

    for x, y, w, h in areas:
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow('find_contours_test', crop_img)
    cv2.waitKey()

    return areas

def distance(a, b):
    dx = (a[0] - b[0]) ** 2
    dy = (a[1] - b[1]) ** 2

    return np.sqrt(dx + dy)

def sort_numbers(numbers):
    sort_y = sorted(numbers, key=lambda r: r[1])

    rows = []
    for i in range(9):
        sliced_row = sort_y[i * 9:(i + 1) * 9]
        sorted_y = sorted(sliced_row, key=lambda r: r[0])
        rows.append(sorted_y)

    return rows

def intersection_area(boxA, boxB):
    left = max(boxA[0], boxB[0])
    top = max(boxA[1], boxB[1])
    right = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    bottom = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if left <= right and top <= bottom:
        return abs((right - left) * (bottom - top))

    return -1

def set_intersections_position(numbers, rows):
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
        recognized_number = model.predict(numbers[number_index][0], k=3)
        sudoku_table[row, column] = recognized_number

    return sudoku_table

def capture_image_or_exist(sudoku_image):

    orig_cropped, crop, rectangle = import_test(sudoku_image)
    thresh_test = filter_test_img(orig_cropped.copy())
    areas = find_contours_test(thresh_test, orig_cropped)

    rows = sort_numbers(areas)

    filt_img, rect = filter_image(crop, rectangle)
    numbers = find_numbers(filt_img, orig_cropped)

    return areas, numbers, rows

def main():
    start_time = time.time()
    cap = cv2.VideoCapture(0)
    sudoku_image = 'sudoku_images/test_1.jpg'

    while True:
        _, image = cap.read()
        cv2.imshow('Image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sudoku_image = image.copy()

            areas, numbers, rows = capture_image_or_exist(sudoku_image)
            print len(areas)
            if len(areas) != 81:
                cv2.destroyAllWindows()
                continue
            else:
                break

        if cv2.waitKey(1) & 0xFF == ord('e'):
            cap.release()
            cv2.destroyAllWindows()
            _, numbers, rows = capture_image_or_exist(sudoku_image)
            break

    knn = KNNClassifier()

    intersections = set_intersections_position(numbers, rows)

    sudoku_table = predict_test_number(numbers, intersections, knn)

    success, steps = sudoku(sudoku_table)
    print "Solved: {0} in {1} steps\n".format(success, steps)
    print "SuDoKu Solver:\n", np.matrix(sudoku_table)
    print ("--- %s seconds ---" % (time.time() - start_time))

    cv2.waitKey()

if __name__ == '__main__':
    main()
