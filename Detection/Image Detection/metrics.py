import cv2
import numpy as np
import glob
import random
import alarm
import time

net = cv2.dnn.readNet("yolov3_training_last (1).weights", "yolov3_testing.cfg")

classes = ["Snake"]

images_path = glob.glob(r"Metrics_Images\*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

frame_id = 0
correct = 0
start = time.time()

for img_path in images_path:
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    frame_id += 1
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, label, (x, y + 30), font, 3, (0, 0, 255), 2)
            correct += 1

    print(frame_id,"finished")
    #img = cv2.resize(img, (500, 300))
    #cv2.imshow("Image", img)
    #key = cv2.waitKey(0)

end = time.time()
print("Total images=",frame_id)
print("No.of images detected correctly=",correct)
acc = (correct/frame_id)*100
print("Acc =",acc)
print("Total time taken =",round(end-start,2))