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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_model_weights(model, model_weights, checkpoint_path, classifier_idx, in_features, device=device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model = model(weights)
    model.classifier[classifier_idx] = nn.Linear(in_features=in_features, out_features=2, bias=True)
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)
    return model 

weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
model = torchvision.models.mobilenet_v3_small
checkpoint_path = "checkpoint_mobile_v3.pth"
model = load_model_weights(model, weights, checkpoint_path, classifier_idx=3, in_features=1024)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.225, 0.224]
    )
])
def empty_or_not(img):
    model.eval()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = test_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.inference_mode():
        y_logits = model(img)
    result = torch.argmax(y_logits, dim=1)
    return result.item()


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
