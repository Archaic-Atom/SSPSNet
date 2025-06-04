import cv2


def resize_half(input_path, output_path):
    # 读取图片
    image = cv2.imread(input_path)

    if image is None:
        print(f"无法读取 {input_path}")
        return

    # 获取原始尺寸
    height, width = image.shape[:2]

    # 计算缩放后的尺寸（原尺寸的一半）
    new_width, new_height = width // 4, height // 4

    # 使用双线性插值缩放
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 保存缩放后的图片
    cv2.imwrite(output_path, resized_image)
    print(f"已保存缩放后的图片: {output_path}")


# 设置图片路径
image_paths = ["/Users/rhc/20250516/left1.jpg",
               "/Users/rhc/20250516/right11.jpg"]  # 替换为你的图片路径
output_paths = ["/Users/rhc/resize_left_000.png", "/Users/rhc/resize_right_000.png"]

# 处理两张图片
for input_path, output_path in zip(image_paths, output_paths):
    resize_half(input_path, output_path)
