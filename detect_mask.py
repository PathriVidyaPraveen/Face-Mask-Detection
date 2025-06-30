import cv2
import torch
import numpy as np
from torchvision import transforms
from model import get_model
from PIL import Image

model = get_model()
model.load_state_dict(torch.load("mask_detector.pth", map_location='cpu'))
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img)  # Convert NumPy to PIL
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            pred = torch.sigmoid(model(face_tensor))[0].item()

        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
