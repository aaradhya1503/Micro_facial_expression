import cv2
import torchvision.transforms as transforms
import torch
import mod

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('v1.mp4')
#cam = cv2.VideoCapture('v2.mp4')
#cam = cv2.VideoCapture('v3.mp4')
cam.set(cv2.CAP_PROP_FPS, 20)

device = mod.get_default_device()
print("Selected device:",device)

w = 'mode.pth'
model = mod.to_device(mod.ATTModel(),device)
if str(device) == 'cpu':
    model.load_state_dict(torch.load(w,map_location=torch.device('cpu')))
if str(device) == 'gpu':
    model.load_state_dict(torch.load(w,map_location=torch.device('cuda')))

transform = transforms.ToTensor()

while True:
    _ , frame = cam.read()
    if _:
        bBox = mod.faceBox(frame)
        if len(bBox) > 0:
            for box in bBox:
                x,y,w,h = box
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                faceExp = frame[y:y+h,x:x+w]
                try: 
                    faceExpResized = cv2.resize(faceExp,(80,80))
                except:
                    continue
                faceExpResizedTensor = transform(faceExpResized)
                prediction = mod.predict_image(faceExpResizedTensor, model, device)
                cv2.putText(frame,prediction,(x,y),cv2.FONT_HERSHEY_COMPLEX,5,(255, 255, 255))
        cv2.imshow('mod', frame)
    if cv2.waitKey(1) & 0xff == ord('q'): 
        print('end')
        break
cam.release()
cv2.destroyAllWindows()