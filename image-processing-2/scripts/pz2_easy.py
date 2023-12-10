import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("image-processing-2/assets/crash.png")

edge_filter = np.array([[-1, -2, -1],
                       [-1,  10, -1],
                       [-1, -2, -1]])

canny_edges = cv2.Canny(image, 50, 150)
filtered_image = cv2.filter2D(image, -1, edge_filter)

plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Оригинал')
plt.subplot(132), plt.imshow(canny_edges, cmap='gray'), plt.title('Фильтр Кэнни')
plt.subplot(133), plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Собственный фильтр')

plt.show()