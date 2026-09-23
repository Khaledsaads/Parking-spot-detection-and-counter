from sklearn.svm import SVC
import numpy as np
import cv2
import pickle
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image
from sklearn.preprocessing import StandardScaler



with open("svc_model.pkl", 'rb') as f:
    svc_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)




def empty_or_not_svc(img):
    resized_img = cv2.resize(img, (15, 15))
    np_img = np.array(resized_img).reshape(1, -1)
    scaled_img = scaler.transform(np_img)
    y_pred = svc_model.predict(scaled_img)
    return y_pred[0]    


def get_parking_spots_bboxes(connected_components: cv2.connectedComponentsWithStats):
    num_labels, labels, stats, centroid = connected_components
    slots= []
    for i in range (1, num_labels):
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        slots.append([x1, y1, w, h])
    return slots
