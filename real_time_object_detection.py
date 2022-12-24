from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2

# Construct the argumet parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, default='frcnn-resnet',
                choices=['frcnn-resnet', 'frcnn-mobilenet', 'retinanet'],
                help='name of the object detection model')
ap.add_argument('-l', '--labels', type=str, default='COCO_classes.txt',
                help='path to list of classes')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='confidence value')
args = vars(ap.parse_args())

# set the device we will be using to run the models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of classes and create colors for bounding boxes
labels = open(args['labels'], 'r').read().split('\n')
CLASSES = labels
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name its torchvision function calls
MODELS = {
    "frcnn-resnet":detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet":detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet":detection.retinanet_resnet50_fpn
}

# load the model and set it to evaluation mode
model = MODELS[args['model']](pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

print("STARTING VIDEO STRAM")

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stram
while True:
    # grab the frame from the thread video and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))

    frame = np.expand_dims(frame, axis=0)
    frame = frame/255.0
    frame = torch.FloatTensor(frame)

    frame = frame.to(DEVICE)
    detections = model(frame)[0]

    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]

        if confidence > args['confidence']:
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx - 1]}: {confidence * 100}"
            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    cv2.imshow("FRAME", orig)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    fps.update()

fps.stop()
print(f"elapsed time :{fps.elapsed()}")
print(f"approx FPS :{fps.fps()}")

cv2.destroyAllWindows()
vs.stop()