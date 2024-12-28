import torch
import torch.nn as nn
import cv2
import mediapipe as mp

class ATTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
    
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(p=0.5),
    
            nn.Flatten(),
            nn.Linear(256*10*10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax())
        
    def forward(self, xb):
        return self.network(xb)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device): 
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
def predict_image(image, model, device):
    xb = to_device(image.unsqueeze(0), device)  
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1) 
    return classes[preds[0].item()] 

findFace = mp.solutions.face_detection.FaceDetection()

def faceBox(frame):
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    height = frame.shape[0]
    width = frame.shape[1]
    results = findFace.process(frameRGB)
    myFaces = []
    if results.detections != None:
        for face in results.detections:
            bBox = face.location_data.relative_bounding_box
            x,y,w,h = int(bBox.xmin*width),int(bBox.ymin*height),int(bBox.width*width),int(bBox.height*height)
            myFaces.append((x,y,w,h))
    return myFaces