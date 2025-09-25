import numpy as np
import cv2

img_path = "2025_09_19__15_12_36.jpg"
label_path = "2025_09_19__15_12_36.txt"

img = cv2.imread(img_path)
h, w, _ = img.shape
points = []
with open(label_path, 'r') as l:
    for line in l:
        parts = line.strip().split()

        if len(parts) == 5:
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        for t in [[-0.5, -0.5],[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]:
            point_x = center_x + t[0] * width
            point_y = center_y + t[1] * height

            point_x = int(point_x * w)
            point_y = int(point_y * h)

            points.append([point_x, point_y])

        cv2.polylines(img, [np.array(points)], True, (255, 255, 255), 4)

    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()