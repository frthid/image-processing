import cv2

video = cv2.VideoCapture(0)

ret, frame1 = video.read()
ret, frame2 = video.read()

min_contour_area = 500

while video.isOpened():
    difference = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(threshold, None, iterations=3)
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (100, 255, 0), 2)

    cv2.imshow("image", frame1)
    frame1 = frame2
    ret, frame2 = video.read()

    if cv2.waitKey(5) == 27:
        break

video.release()
cv2.destroyAllWindows()
