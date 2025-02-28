import cv2
import numpy as np


def reconstruct_right_image(left_image_path, disparity_map_path, baseline=0.54, focal_length=721.5):
    # 读取左图和视差图
    left_image = cv2.imread(left_image_path)
    disparity_map = cv2.imread(disparity_map_path, cv2.IMREAD_UNCHANGED)  # 读取uint16类型的视差图

    h, w = disparity_map.shape
    right_image = np.zeros_like(left_image)

    for y in range(h):
        for x in range(w):
            disparity = disparity_map[y, x] / 256.0  # 视差值通常存储为 uint16，需要缩放
            new_x = int(x - disparity)

            if 0 <= new_x < w:
                right_image[y, new_x] = left_image[y, x]

    return right_image


# 示例使用
left_image_path = "/Users/rhc/resize_left_000.png"
disparity_map_path = "/Users/rhc/Downloads/resize_left_000_uaxGSBN.png"
right_image = reconstruct_right_image(left_image_path, disparity_map_path)
cv2.imwrite("Users/rhc/reconstructed_right.png", right_image)
cv2.imshow("Right Image", right_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
