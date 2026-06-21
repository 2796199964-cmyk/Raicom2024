import cv2
import os
import numpy as np
from random import sample

# 设置源目录和目标目录
src_dir = './'
dst_dir = 'result'


# 如果目标目录不存在，则创建它
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 获取源目录下所有jpg文件的路径
jpg_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.jpg')]

# 随机选择文件
files_to_process = jpg_files if len(jpg_files) <= 10 else sample(jpg_files, 10)

for filepath in files_to_process:
    filename = os.path.basename(filepath)
    # 读取图像
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Unable to load image {filename}")
        continue

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用阈值处理来识别黑色区域
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # 找到面积最大的轮廓
    c = max(contours, key=cv2.contourArea) if contours else None

    # 如果找到了轮廓，使用以下方法找到轮廓上距离最远的两点
    if c is not None:
        # 获取轮廓的凸包
        hull = cv2.convexHull(c, returnPoints=True)

        # 使用红色线条在原图上绘制凸包
        cv2.drawContours(img, [hull], 0, (0, 255, 255), 20)

        # 初始化两点和最大距离
        point1, point2 = hull[0], hull[1]
        max_distance = cv2.norm(point1, point2)

        # 遍历凸包上的所有点对，找到距离最远的两点
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                dist = cv2.norm(hull[i], hull[j])
                if dist > max_distance:
                    max_distance = dist
                    point1, point2 = hull[i], hull[j]

        # 确定哪一点是上方的点，哪一点是下方的点
        if point1[0][1] < point2[0][1]:  # y坐标较小的是上方的点
            upper_point, lower_point = point1[0], point2[0]
        else:
            upper_point, lower_point = point2[0], point1[0]

        # 以下方的点的y坐标为水平线，找到与凸包的另一侧交点
        horizontal_y = lower_point[1]
        new_lower_point = None
        intersection_dist = 0

        # 遍历凸包的每条边，找到与水平线的交点
        for i in range(len(hull)):
            p1 = hull[i][0]
            p2 = hull[(i + 1) % len(hull)][0]  # 循环到下一点，形成闭合凸包

            # 检查边是否跨越水平线
            if (p1[1] <= horizontal_y and p2[1] >= horizontal_y) or \
                    (p1[1] >= horizontal_y and p2[1] <= horizontal_y):
                # 计算交点
                if p2[1] != p1[1]:  # 避免除以零
                    t = (horizontal_y - p1[1]) / (p2[1] - p1[1])
                    intersect_x = p1[0] + t * (p2[0] - p1[0])

                    # 计算交点到当前下方点的距离
                    temp_dist = abs(intersect_x - lower_point[0])

                    # 选择距离当前下方点最远的交点（即凸包的另一侧）
                    if temp_dist > intersection_dist:
                        intersection_dist = temp_dist
                        new_lower_point = (int(intersect_x), horizontal_y)

        # 如果找到了新的下方点，使用它
        if new_lower_point is not None:
            lower_point = new_lower_point
        else:
            # 如果没有找到合适的交点，使用原始的下方点
            lower_point = (lower_point[0], lower_point[1])

        # 确保point1是上方的点，point2是修改后的下方点
        point1 = (int(upper_point[0]), int(upper_point[1]))
        point2 = (int(lower_point[0]), int(lower_point[1]))


        # 计算斜率
        if point1[0] == point2[0]:  # 防止除以零
            slope = 'infinity'
            angle = 0.0  # 垂直线
        else:
            # 计算斜率（注意OpenCV坐标系的y轴方向）
            slope = -(point2[1] - point1[1]) / (point2[0] - point1[0])

            # 斜率对应的角度（弧度），然后转换为度
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)

            # 调整角度：使竖直向上为0°，向右为正，向左为负
            angle = angle_deg - 90
            angle = -angle

            # 标准化角度到(-90, 90]范围
            if angle > 90:
                angle -= 180
            elif angle <= -90:
                angle += 180

        # 输出角度信息
        if slope != 'infinity':
            if abs(angle) < 3:
                direction = "straight"
            elif angle > 0:
                direction = "right"
            elif angle < 0:
                direction = "left"
            print(f" {filename} : 角度={angle:.2f}° 方向={direction}")
        else:
            direction = "straight"
            angle = 0.0
            print(f" {filename} : 角度={angle:.2f}° 方向={direction}")

        # 修改文件名，加入角度信息
        if isinstance(slope, float):  # 如果斜率是浮点数
            slope_str = f"{slope:.2f}"  # 格式化为两位小数的字符串
        else:  # 斜率是'infinity'
            slope_str = slope

        # 在结果文件名中加入角度信息
        result_filename = f"{direction}_{os.path.splitext(filename)[0]}({angle:.1f}){os.path.splitext(filename)[1]}"

        # 绘制直线
        cv2.line(img, point1, point2, (0, 0, 255), 20)

    # 保存结果图像
    cv2.imwrite(os.path.join(dst_dir, result_filename), img)
