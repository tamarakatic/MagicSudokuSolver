import cv2
import numpy as np
from knn_dataset import create_dataset, img_resize

class KNNClassifier:

    def __init__(self):

        print "\n=> K-Nearest Neighbors ..."
        print "=> Creating KNN classifier model ..."

        training_img, training_label = create_dataset('dataset/*.jpg')
        self.model = cv2.ml.KNearest_create()
        self.model.train(training_img, cv2.ml.ROW_SAMPLE, training_label)


    def predict(self, test_img, k=3):
        resize_img = test_img.reshape(-1, 28*28).astype(np.float32)
        returnVel, result, neighbors, dist = self.model.findNearest(resize_img, k)
        return result[0][0]
