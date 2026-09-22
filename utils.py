from sklearn.svm import SVC
import numpy as np
import cv2
import pickle


def load_model():
    with open('svc_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model 


def empty_or_not(img):
    model = load_model()
    resized_img = cv2.resize(img, (15, 15))
    arr_img = np.array(resized_img).flatten()
    arr_img = arr_img.reshape(1, -1)
    result = model.predict(arr_img)
    return result


def get_parking_spots_bboxes(connected_components: cv2.connectedComponentsWithStats):
    num_labels, labels, stats, centroid = connected_components
    slots= []
    for i in range (1, len(labels)):
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        slots.append([x1, y1, w, h])
    return slots