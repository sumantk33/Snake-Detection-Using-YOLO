import cv2
import yolo_snake_detection

cap = cv2.VideoCapture('clip2.mp4')

yolo_snake_detection.detect_snake(cap)
