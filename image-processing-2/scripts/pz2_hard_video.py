import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    min_threshold = 100
    max_threshold = 200
    binary_mask = cv2.inRange(v, min_threshold, max_threshold)
    segmented_object = cv2.bitwise_and(frame, frame, mask=binary_mask)
    blurred_image = cv2.GaussianBlur(v, (49, 49), 0)
    edges_subtraction = v - blurred_image
    edges_canny = cv2.Canny(v, 50, 150)

    cv2.imshow('Оригинал', frame)
    cv2.imshow('Сегментированный объект', segmented_object)
    cv2.imshow('C выделением границ вычитанием', edges_subtraction)
    cv2.imshow('Фильтр Canny', edges_canny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()