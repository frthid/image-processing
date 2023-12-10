import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "image-processing-2/assets/crash.png"


original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image)
fig, ax = plt.subplots(figsize=(12, 6))
hist_color = ['red', 'green', 'blue']
hist_label = ['red', 'green', 'blue']
for i, channel in enumerate([h, s, v]):
    ax.hist(channel.ravel(), bins=256, color=hist_color[i], alpha=0.5, rwidth=0.8, label=hist_label[i])
    
ax.legend()
plt.show()

min_threshold = 100
max_threshold = 200

binary_mask = cv2.inRange(v, min_threshold, max_threshold)
segmented_object = cv2.bitwise_and(original_image_rgb, original_image_rgb, mask=binary_mask)

plt.figure(figsize=(8, 8))
plt.imshow(segmented_object)
plt.title('Сегментированный объект')
plt.show()

#Выделения границ с помощью вычитания
blurred_image = cv2.GaussianBlur(v, (49, 49), 0)
edges_subtraction = v - blurred_image
edges_canny = cv2.Canny(v, 50, 150)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(v, cmap='gray'), plt.title('Оригинал')
plt.subplot(132), plt.imshow(edges_subtraction, cmap='gray'), plt.title('C выделением границ вычитанием')
plt.subplot(133), plt.imshow(edges_canny, cmap='gray'), plt.title('Фильтр Canny')

plt.show()