import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(1)  # You can change to 1 if using external cam

if not cap.isOpened():
    raise IOError("Cannot access webcam")

cv2.namedWindow('Adjustments')
cv2.resizeWindow('Adjustments', 500, 375)

# Trackbars
cv2.createTrackbar('H Min', 'Adjustments', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Adjustments', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Adjustments', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Adjustments', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Adjustments', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Adjustments', 255, 255, nothing)
cv2.createTrackbar('Min Contour Area', 'Adjustments', 1000, 10000, nothing)
cv2.createTrackbar('Polygonal Sides', 'Adjustments', 3, 10, nothing)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get values from trackbars
        h_min = cv2.getTrackbarPos('H Min', 'Adjustments')
        h_max = cv2.getTrackbarPos('H Max', 'Adjustments')
        s_min = cv2.getTrackbarPos('S Min', 'Adjustments')
        s_max = cv2.getTrackbarPos('S Max', 'Adjustments')
        v_min = cv2.getTrackbarPos('V Min', 'Adjustments')
        v_max = cv2.getTrackbarPos('V Max', 'Adjustments')
        min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Adjustments')
        sides = cv2.getTrackbarPos('Polygonal Sides', 'Adjustments')

        lower_hsv = (h_min, s_min, v_min)
        upper_hsv = (h_max, s_max, v_max)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_frame = np.zeros_like(frame)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == sides and cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.drawContours(contour_frame, [contour], -1, (0, 255, 255), 2)

        # Resize all for grid display
        def rescale(img):
            return cv2.resize(img, (320, 240))

        top_row = np.hstack((rescale(frame), rescale(result)))
        bottom_row = np.hstack((rescale(contour_frame), rescale(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow('Four-Box View', grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
