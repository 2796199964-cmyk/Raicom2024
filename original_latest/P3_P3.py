import numpy as np
import easyocr
import cv2
import os

import sys
print(sys.version)

# 初始化reader，指定要识别的语言
reader = easyocr.Reader(['ch_sim'])

# 文件夹路径
folder_path = 'd:\\Desktop\\Raicom\\R\\P3\\'

# 获取文件夹中所有的.jpg文件
selected_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 遍历选中的文件并处理
for file_name in selected_files:
    # 图片路径
    image_path = os.path.join(folder_path, file_name)

    # 加载图片
    img = cv2.imread(image_path)

    # 使用easyocr识别文本
    results = reader.readtext(img)
    for (bbox, text, prob) in results:
        if "大厦" in text:
            head = text

    # 转为HSV颜色空间，便于处理颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义黑色的HSV范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    # 定义红黄色的HSV范围
    lower_red_yellow = np.array([0, 100, 100])
    upper_red_yellow = np.array([30, 255, 255])

    # 创建黑色掩模
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 创建红黄色掩模
    mask_red_yellow = cv2.inRange(hsv, lower_red_yellow, upper_red_yellow)

    # 对掩模进行形态学处理，去除噪点
    kernel = np.ones((10, 10), np.uint8)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    mask_red_yellow = cv2.morphologyEx(mask_red_yellow, cv2.MORPH_OPEN, kernel)

    # 查找黑色的轮廓
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选出面积大于特定阈值的黑色轮廓
    min_area_black = 5000
    large_contours_black = [cnt for cnt in contours_black if cv2.contourArea(cnt) > min_area_black]

    # 初始化用于记录层次的列表
    layer_ys = []
    contours_by_layer = {}

    # 在原图上用外接矩形标记大的黑色区域，并计算层次
    height, width, _ = img.shape

    for contour in large_contours_black:
        x, y, w, h = cv2.boundingRect(contour)
        if x + w >= width - 100:
            continue

        if x > 100 and y + h < height - 50 and w < 150 and h < 300 and h > 80:
            # 将轮廓添加到相应层次的列表中
            added_to_layer = False
            for layer_y in layer_ys:
                if abs(y - layer_y) < 60:
                    if layer_y not in contours_by_layer:
                        contours_by_layer[layer_y] = []
                    contours_by_layer[layer_y].append(contour)
                    added_to_layer = True
                    break
            if not added_to_layer:
                layer_ys.append(y)
                contours_by_layer[y] = [contour]

    # 按y坐标值从大到小排序
    sorted_layers = sorted(contours_by_layer.items(), key=lambda item: item[0], reverse=True)

    # 记录从上往下的每个绿色框的下边界y坐标
    layer_bottoms = {}
    for i, (layer_y, contours) in enumerate(sorted_layers):
        # 合并轮廓
        all_points = np.vstack(contours).astype(np.int32)
        x, y, w, h = cv2.boundingRect(all_points)

        # 记录该层的下边界y坐标
        layer_bottoms[layer_y] = y + h

        # 画出外接矩形
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)

    # 在miny和maxy之间查找红黄色块
    contours_red_yellow, _ = cv2.findContours(mask_red_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选出所有红黄色轮廓，并设定一个列表存储黄色框所在楼层
    yellow_boxes_layers = []
    for contour in contours_red_yellow:
        x, y, w, h = cv2.boundingRect(contour)

        # 找到黄色框所在的绿色框编号
        for j, (layer_y, bottom_y) in enumerate(sorted(layer_bottoms.items(), key=lambda item: item[1])):
            if y < bottom_y:
                yellow_boxes_layers.append(j + 1) # j+1 是从上往下数的第几层
                # 画出黄色框
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 10)
                break

    # --- 核心修改点 ---
    # 将 "从上往下数的楼层" 转换为 "从下往上数的楼层"
    total_floors = len(layer_ys)
    # 使用列表推导式对 yellow_boxes_layers 中的每个元素进行转换
    yellow_boxes_layers = [total_floors - floor_index_from_top + 1 for floor_index_from_top in yellow_boxes_layers]

    # 输出黄色框所在的楼层信息
    if yellow_boxes_layers:
        print(f"图片{file_name}中楼宇为{head},火灾发生在：{'层、'.join(map(str, yellow_boxes_layers))}层。")
    elif 1:
        print(f"图片{file_name}中楼宇为{head},没有发生火灾。")

    # 新图片保存路径
    output_folder = 'd:\\Desktop\\Raicom\\R\\P3\\result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_image_path = os.path.join(output_folder, f'{len(layer_ys)}_{file_name}')

    # 保存处理后的图像
    cv2.imwrite(output_image_path, img)