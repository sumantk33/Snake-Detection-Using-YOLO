import cv2
import numpy as np
import glob
import random
import time
import alarm

def detect_snake(cap):
    net = cv2.dnn.readNet("yolov3_training_last (1).weights", "yolov3_testing.cfg")

    classes = ["snake"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN

    start = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        frame_id += 1
        detected = 0

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

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
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                color = (0,0,255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)
                detected = 1
                
        elapsed_time = time.time() - start
        fps = frame_id/elapsed_time
        frame = cv2.resize(frame, (500,300))
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10,30), font, 2, (0, 255, 0), 1)
        cv2.imshow("Image", frame)
        if detected:
            alarm.play_alarm()
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
