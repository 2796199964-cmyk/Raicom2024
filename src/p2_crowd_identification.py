import os
import random
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

CONFIG = {
    "model_path": os.environ.get("P2_MODEL_PATH", "models/yolov8l-pose.pt"),
    "input_folder": os.environ.get("P2_INPUT_DIR", "data/P2"),
    "output_folder": os.environ.get("P2_OUTPUT_DIR", "outputs/P2"),
    "sample_size": 200,
    "upper_body_indices": [5, 6, 7, 8],  # COCO: 左肩, 右肩, 左肘, 右肘
}

def ensure_output_dir(output_path: str):
    """确保输出目录存在"""
    os.makedirs(output_path, exist_ok=True)

# === 移除了不再使用的 get_upper_body_bbox 函数 ===

def is_red_blue_or_gray_from_hsv(avg_h, avg_s, avg_v):
    """
    根据平均HSV值判断颜色。
    OpenCV的HSV格式: H(0-179), S(0-255), V(0-255)
    """
    s_norm = avg_s / 255.0
    v_norm = avg_v / 255.0

    if v_norm < 0.4 or s_norm < 0.4:
        return "灰色系"

    if (0 <= avg_h < 18) or (160 <= avg_h <= 179):
        return "红色系"
    elif 80 <= avg_h < 140:
        return "蓝色系"
    else:
        return "其他"

def analyze_person_color(source_hsv, keypoint_set, indices):
    """
    分析一个人的上衣主色。
    策略：统计4个关键点3x3区���内的所有像素颜色，
          若"灰色系"与某非灰色颜色票数相同，则优先选择非灰色颜色。
    """
    h, w = source_hsv.shape[:2]
    all_pixel_colors = []

    for i in indices:
        try:
            cx, cy = int(keypoint_set[i][0]), int(keypoint_set[i][1])
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            x_start = max(0, cx - 1)
            x_end = min(w, cx + 2)
            y_start = max(0, cy - 1)
            y_end = min(h, cy + 2)

            patch_hsv = source_hsv[y_start:y_end, x_start:x_end]
            for row in patch_hsv:
                for pixel_hsv in row:
                    h_val, s_val, v_val = pixel_hsv
                    pixel_color = is_red_blue_or_gray_from_hsv(h_val, s_val, v_val)
                    all_pixel_colors.append(pixel_color)

        except (IndexError, ValueError):
            continue

    if not all_pixel_colors:
        return "其他"

    color_counter = Counter(all_pixel_colors)
    most_common = color_counter.most_common()

    # 情况1: 只有一种颜色
    if len(most_common) == 1:
        return most_common[0][0]

    # 获取最高票数
    max_count = most_common[0][1]

    # 找出所有得票等于 max_count 的颜色
    top_colors = [color for color, count in most_common if count == max_count]

    # 如果"灰色系"在 top_colors 中，且还有其他非灰色颜色 → 排除灰色
    if "灰色系" in top_colors and len(top_colors) > 1:
        # 优先选择非灰色的颜色（按红、蓝顺序）
        for candidate in ["红色系", "蓝色系"]:
            if candidate in top_colors:
                return candidate
        # 如果没有红/蓝，但有"其他"，也优先于灰色
        if "其他" in top_colors:
            return "其他"
        # 否则 fallback 到第一个非灰色（理论上不会走到这里）
        for color in top_colors:
            if color != "灰色系":
                return color

    # 默认：返回得票最高的颜色（包括灰色单独最高时）
    return most_common[0][0]

def draw_keypoints_with_hsv_and_label(image, keypoint_set, indices, source_hsv, color_label, config):
    """
    在四个关键点上画点，并用PIL在每个点旁边添加文字。
    同时，绘制连接四个关键点的多边形，并在多边形内部标记主色。
    所有绘图都在PIL图像上完成。
    """
    h, w = image.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("simhei.ttf", 24)
    except OSError:
        try:
            font = ImageFont.truetype("msyh.ttc", 24)
        except OSError:
            font = ImageFont.load_default()

    # 遍历四个关键点
    valid_points = []  # 收集有效的点用于绘制多边形
    for i in indices:
        try:
            x, y = int(keypoint_set[i][0]), int(keypoint_set[i][1])
            if not (0 <= x < w and 0 <= y < h and x > 0 and y > 0):
                continue

            # === 1. 绘制关键点 (在PIL上画) ===
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255), outline=(255, 255, 255))

            # 获取该点的HSV值
            h_val, s_val, v_val = source_hsv[y, x]

            # 判断该点的颜色
            point_color_label = is_red_blue_or_gray_from_hsv(h_val, s_val, v_val)

            # === 2. 只在关键点旁显示 point_color_label ===
            text_x = x + 5
            text_y = y - 5
            draw.text((text_x, text_y), point_color_label, font=font, fill=( 0, 255, 0))

            # 将有效点加入列表
            valid_points.append([x, y])

        except (IndexError, ValueError):
            continue

    # === 3. 绘制多边形并在其内部标记主色 ===
    if len(valid_points) >= 4:
        # 按照左肩 -> 左肘 -> 右肘 -> 右肩的顺序连接
        polygon_points = [
            tuple(valid_points[0]),  # 左肩 (index 5)
            tuple(valid_points[2]),  # 左肘 (index 7)
            tuple(valid_points[3]),  # 右肘 (index 8)
            tuple(valid_points[1])  # 右肩 (index 6)
        ]
        # 绘制多边形框
        draw.polygon(polygon_points, outline=(0, 255, 0), width=3)

        # 计算多边形的中心点，用于放置主色标签
        center_x = sum(p[0] for p in polygon_points) // len(polygon_points)
        center_y = sum(p[1] for p in polygon_points) // len(polygon_points)

        # 在中心点绘制主色标签 {color_label}
        draw.text((center_x - 20, center_y - 10), color_label, font=font, fill=(0, 255, 0))

    # 将PIL图像转回OpenCV BGR格式
    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def create_full_body_mask_from_keypoints(keypoints, img_shape):
    """
    从17个COCO关键点创建一个全身的凸包掩码。
    """
    h, w = img_shape[:2]
    # 过滤掉无效的关键点 (坐标 <= 0)
    valid_points = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x > 0 and y > 0 and x < w and y < h:
            valid_points.append([x, y])

    if len(valid_points) < 3:
        return None

    # 创建凸包 (Convex Hull) 作为全身区域
    points_np = np.array(valid_points, dtype=np.int32)
    hull = cv2.convexHull(points_np)

    # 创建掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    return mask


def create_dilated_full_body_mask_from_keypoints(keypoints, img_shape, dilate_kernel_size=30):
    """
    从17个COCO关键点创建一个全身的凸包掩码，并对其进行膨胀操作。

    Args:
        keypoints: (17, 2) 的 numpy 数组，包含关键点坐标。
        img_shape: 图像的形状 (H, W, C)。
        dilate_kernel_size: 膨胀核的大小。值越大，膨胀范围越广。
                            对于 1920x1080 的图像，建议从 25-40 开始尝试。

    Returns:
        膨胀后的二值掩码 (H, W)，类型为 uint8，前景为255，背景为0。
    """
    h, w = img_shape[:2]
    # 过滤掉无效的关键点 (坐标 <= 0)
    valid_points = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x > 0 and y > 0 and x < w and y < h:
            valid_points.append([x, y])

    if len(valid_points) < 3:
        return None

    # 1. 创建凸包 (Convex Hull) 作为全身区域
    points_np = np.array(valid_points, dtype=np.int32)
    hull = cv2.convexHull(points_np)

    # 创建基础掩码
    base_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(base_mask, [hull], 255)

    # 2. === 关键步骤：对基础掩码进行膨胀 ===
    # 定义一个椭圆形的结构元素（核），这比矩形核更自然
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilate_kernel_size, dilate_kernel_size)
    )
    # 执行膨胀操作
    dilated_mask = cv2.dilate(base_mask, kernel, iterations=1)

    return dilated_mask


def draw_dilated_hull_on_image(image, keypoints, dilate_kernel_size=30):
    """
    在给定的图像上，根据关键点绘制一个膨胀后的凸包轮廓（红色）。

    Args:
        image: OpenCV BGR 格式的图像 (H, W, C)。
        keypoints: (17, 2) 的 numpy 数组，包含关键点坐标。
        dilate_kernel_size: 膨胀核的大小。
    """
    h, w = image.shape[:2]
    valid_points = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x > 0 and y > 0 and x < w and y < h:
            valid_points.append([x, y])

    if len(valid_points) < 3:
        return

    # 1. 创建凸包
    points_np = np.array(valid_points, dtype=np.int32)
    hull = cv2.convexHull(points_np)

    # 2. 创建基础掩码
    base_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(base_mask, [hull], 255)

    # 3. 膨胀掩码
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilate_kernel_size, dilate_kernel_size)
    )
    dilated_mask = cv2.dilate(base_mask, kernel, iterations=1)

    # 4. 找到膨胀后掩码的轮廓
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 绘制最大的轮廓（通常只有一个）
        cv2.drawContours(image, contours, -1, color=(0, 0, 255), thickness=2)  # BGR: 红色


def process_image(model, img_path: str, output_dir: str, config: dict):
    filename = os.path.basename(img_path)
    source = cv2.imread(img_path)
    if source is None:
        print(f"警告：无法读取图像 {img_path}")
        return filename, 0, 0, 0, 0

    h, w = source.shape[:2]
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

    # === 核心：创建一个用于检测的工作图像副本 ===
    working_img = source.copy()

    angles_to_try = [0, 45, -45, 90, -90]
    final_unique_keypoints = []

    # 我们将循环多次，直到所有角度都检测不到新的人为止
    max_iterations = 10  # 防止无限循环
    for iteration in range(max_iterations):
        found_new_person = False
        all_keypoints_this_round = []

        # 对每个角度，在当前的 working_img 上进行检测
        for angle in angles_to_try:
            if angle == 0:
                rotated_img = working_img.copy()
                rot_mat = None
            else:
                center = (w // 2, h // 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos_val = np.abs(rot_mat[0, 0])
                sin_val = np.abs(rot_mat[0, 1])
                new_w = int((h * sin_val) + (w * cos_val))
                new_h = int((h * cos_val) + (w * sin_val))
                rot_mat[0, 2] += (new_w / 2) - center[0]
                rot_mat[1, 2] += (new_h / 2) - center[1]
                rotated_img = cv2.warpAffine(working_img, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

            results = model(rotated_img)
            if not results or len(results[0].keypoints.xy) == 0:
                continue

            result = results[0]
            detected_keypoints = result.keypoints.xy.cpu().numpy()  # (N, 17, 2)

            # 将关键点反变换回原始坐标系
            if angle == 0:
                keypoints_in_original = detected_keypoints
            else:
                inv_rot_mat = cv2.invertAffineTransform(rot_mat)
                flat_kps = detected_keypoints.reshape(-1, 2)
                ones = np.ones((flat_kps.shape[0], 1))
                flat_kps_homogeneous = np.hstack([flat_kps, ones])
                transformed_flat = (inv_rot_mat @ flat_kps_homogeneous.T).T
                keypoints_in_original = transformed_flat.reshape(detected_keypoints.shape)

            all_keypoints_this_round.extend(keypoints_in_original)

        # 如果这一轮没检测到任何人，就退出
        if not all_keypoints_this_round:
            break

        # 处理这一轮检测到的所有人
        for keypoint_set in all_keypoints_this_round:
            # 检查这个人的关键点是否大部分落在了已被覆盖（白色）的区域
            # 简单策略：检查鼻子（index 0）或肩膀是否有效
            nose_x, nose_y = keypoint_set[0]
            if nose_x <= 0 or nose_y <= 0:
                continue

            # 检查工作图像上该点是否是白色（已被覆盖）
            b, g, r = working_img[int(nose_y), int(nose_x)]
            if b == 255 and g == 255 and r == 255:
                # 这个点在白色区域，很可能是重复检测，跳过
                continue

            # === 找到了一个新的人！ 使用膨胀后的掩码 ===
            mask = create_dilated_full_body_mask_from_keypoints(
                keypoint_set,
                working_img.shape,
                dilate_kernel_size=35  # <-- 膨胀值修改
            )
            if mask is not None:
                # 将掩码区域设为白色
                working_img[mask == 255] = [255, 255, 255]

            # === 关键修复：将这个人的关键点加入最终列表 ===
            final_unique_keypoints.append(keypoint_set)
            
            # === 修复：标记找到了新的人 ===
            found_new_person = True


        # 如果这一轮没有找到新的人，也退出
        if not found_new_person:
            break

    # === 对最终去重后的人进行分析和绘制（在原始图像source上操作）===
    person_num = red = blue = gray = 0
    indices = config["upper_body_indices"]

    for keypoint_set in final_unique_keypoints:
        main_color = analyze_person_color(source_hsv, keypoint_set, indices)
        if main_color == "红色系":
            red += 1
        elif main_color == "蓝色系":
            blue += 1
        elif main_color == "灰色系":
            gray += 1

        draw_keypoints_with_hsv_and_label(
            source,
            keypoint_set,
            indices,
            source_hsv=source_hsv,
            color_label=main_color,
            config=config
        )

        # === 新增：绘制红色的膨胀凸包轮廓 ===
        draw_dilated_hull_on_image(
            source,
            keypoint_set,
            dilate_kernel_size=35
        )

    person_num = len(final_unique_keypoints)

    # === 绘制统计信息 ===
    pil_img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    text = f"总人数{person_num}人；红色系{red}人，蓝色系{blue}人，灰色系{gray}人"
    try:
        font = ImageFont.truetype("simhei.ttf", 24)
    except OSError:
        try:
            font = ImageFont.truetype("msyh.ttc", 24)
        except OSError:
            font = ImageFont.load_default()

    img_width, _ = pil_img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_x = img_width - (bbox[2] - bbox[0]) - 10
    draw.text((text_x, 10), text, font=font, fill=(255, 0, 255))

    source = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, source)

    return filename, person_num, red, blue, gray


def main():
    cfg = CONFIG
    model = YOLO(cfg["model_path"])
    input_folder = cfg["input_folder"]
    output_folder = cfg["output_folder"]
    ensure_output_dir(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    if not image_files:
        print("错误：输入文件夹中无 .jpg 图像！")
        return

    sample_size = min(cfg["sample_size"], len(image_files))
    selected_images = random.sample(image_files, sample_size)
    print(f"共找到 {len(image_files)} 张图像，随机选取 {sample_size} 张进行处理...")

    results_list = []
    for img_file in selected_images:
        img_path = os.path.join(input_folder, img_file)
        result = process_image(model, img_path, output_folder, cfg)
        results_list.append(result)

    print("\n" + "="*80)
    print("检测结果汇总：")
    print("="*80)
    for img_name, num_persons, red, blue, gray in results_list:
        print(f"图片"{img_name}"：总人数{num_persons}人；红色系{red}人，蓝色系{blue}人，灰色系{gray}人")

if __name__ == "__main__":
    main()
