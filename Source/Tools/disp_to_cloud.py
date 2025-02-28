import cv2
import numpy as np


def disparity_to_pointcloud_with_color(left_image_path, disparity_path, f, B, output_xyzrgb_path):
    """
    将视差图转换为带颜色的点云（XYZRGB格式）

    参数:
        left_image_path: 左侧图像路径（用于坐标计算）
        disparity_path: 视差图路径
        f: 相机焦距
        B: 相机基线
        color_image_path: 彩色图像路径（用于颜色提取）
        output_xyzrgb_path: 输出XYZRGB文件路径

    返回:
        None
    """
    # 读取图像
    left_img = cv2.imread(left_image_path, 0)  # 灰度左图
    color_img = cv2.imread(left_image_path)   # 彩色图像
    disparity = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH) / 256  # 16位视差图

    h, w = left_img.shape
    cx = w / 2.0  # 光心x坐标
    cy = h / 2.0  # 光心y坐标

    points = []

    for v in range(h):
        for u in range(w):
            d = disparity[v][u]
            if d == 0:
                continue  # 跳过无效像素

            # 计算深度z
            z = (f * B) / d

            # 计算三维坐标
            x = (u - cx) * z / f
            y = (v - cy) * z / f

            # 获取颜色值（确保坐标在图像范围内）
            if 0 <= u < w and 0 <= v < h:
                r, g, b = color_img[v, u]  # OpenCV读取的BGR格式
                points.append((x, y, z, r, g, b))

    # 保存为XYZRGB文件
    with open(output_xyzrgb_path, 'w') as f:
        for p in points:
            f.write(f'{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n')


def main() -> int:

    # 示例调用
    disparity_to_pointcloud_with_color(
        '/Users/rhc/0001_L.png',
        # '/Users/rhc/3_720/images/left_000.png',       # 左图路径
        '/Users/rhc/0001_L.png',   # 视差图路径
        f=521.3689575195312,          # 焦距（需根据实际标定修改）
        B=120.02717590332031,           # 基线（需根据实际标定修改）
        output_xyzrgb_path='/Users/rhc/1.txt'
    )

    return 0


if __name__ == "__main__":
    main()
