import cv2
import numpy as np

base_path = "image-processing-3/assets/"
laser_centers = []

def find_point_center(image_name):
    image = cv2.imread(base_path + image_name, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_area_contour_index = -1

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_area_contour_index = i

    bounding_box = cv2.boundingRect(contours[max_area_contour_index])
    center = (bounding_box[0] + bounding_box[2] // 2, bounding_box[1] + bounding_box[3] // 2)
    laser_centers.append(center)
    print(f"Центр лазера: X = {center[0]}, Y = {center[1]}")

    cv2.rectangle(image, bounding_box, (0, 255, 0), 2)
    cv2.circle(image, center, 1, (0, 0, 255), 2)
    cv2.imshow(f"{image_name}_result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_names = [f"img_{i+1}.jpeg" for i in range(8)]

for name in image_names:
    find_point_center(name)

if len(laser_centers) > 1:
    mean = np.mean(laser_centers, axis=0)
    dispersion = np.var(laser_centers, axis=0)
    print(f"Дисперсия центра лазера: X = {dispersion[0]}, Y = {dispersion[1]}")
    std_dev = np.sqrt(dispersion)
    print(f"СКО центра лазера: X = {std_dev[0]}, Y = {std_dev[1]}")
else:
    print("Недостаточно данных для вычисления дисперсии")

