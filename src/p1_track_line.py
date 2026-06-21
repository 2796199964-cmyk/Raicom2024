import argparse
import os
from random import sample

import cv2
import numpy as np


def process_folder(input_dir: str, output_dir: str, sample_size: int = 10) -> None:
    os.makedirs(output_dir, exist_ok=True)
    jpg_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".jpg")
    ]
    files_to_process = jpg_files if len(jpg_files) <= sample_size else sample(jpg_files, sample_size)

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error: Unable to load image {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        c = max(contours, key=cv2.contourArea) if contours else None

        direction = "unknown"
        angle = 0.0
        result_filename = filename

        if c is not None:
            hull = cv2.convexHull(c, returnPoints=True)
            cv2.drawContours(img, [hull], 0, (0, 255, 255), 20)

            point1, point2 = hull[0], hull[1]
            max_distance = cv2.norm(point1, point2)
            for i in range(len(hull)):
                for j in range(i + 1, len(hull)):
                    dist = cv2.norm(hull[i], hull[j])
                    if dist > max_distance:
                        max_distance = dist
                        point1, point2 = hull[i], hull[j]

            if point1[0][1] < point2[0][1]:
                upper_point, lower_point = point1[0], point2[0]
            else:
                upper_point, lower_point = point2[0], point1[0]

            horizontal_y = lower_point[1]
            new_lower_point = None
            intersection_dist = 0

            for i in range(len(hull)):
                p1 = hull[i][0]
                p2 = hull[(i + 1) % len(hull)][0]
                if (p1[1] <= horizontal_y <= p2[1]) or (p2[1] <= horizontal_y <= p1[1]):
                    if p2[1] != p1[1]:
                        t = (horizontal_y - p1[1]) / (p2[1] - p1[1])
                        intersect_x = p1[0] + t * (p2[0] - p1[0])
                        temp_dist = abs(intersect_x - lower_point[0])
                        if temp_dist > intersection_dist:
                            intersection_dist = temp_dist
                            new_lower_point = (int(intersect_x), int(horizontal_y))

            if new_lower_point is not None:
                lower_point = new_lower_point
            else:
                lower_point = (int(lower_point[0]), int(lower_point[1]))

            point1 = (int(upper_point[0]), int(upper_point[1]))
            point2 = (int(lower_point[0]), int(lower_point[1]))

            if point1[0] == point2[0]:
                slope = "infinity"
                angle = 0.0
            else:
                slope = -(point2[1] - point1[1]) / (point2[0] - point1[0])
                angle_deg = np.degrees(np.arctan(slope))
                angle = -(angle_deg - 90)
                if angle > 90:
                    angle -= 180
                elif angle <= -90:
                    angle += 180

            if slope != "infinity":
                if abs(angle) < 3:
                    direction = "straight"
                elif angle > 0:
                    direction = "right"
                else:
                    direction = "left"
            else:
                direction = "straight"

            print(f"{filename}: 角度={angle:.2f}° 方向={direction}")
            stem, ext = os.path.splitext(filename)
            result_filename = f"{direction}_{stem}({angle:.1f}){ext}"
            cv2.line(img, point1, point2, (0, 0, 255), 20)

        cv2.imwrite(os.path.join(output_dir, result_filename), img)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 轨道线方向识别")
    parser.add_argument("--input", default="data/P1", help="P1 图片目录")
    parser.add_argument("--output", default="outputs/P1", help="结果输出目录")
    parser.add_argument("--sample-size", type=int, default=10, help="随机抽样数量")
    args = parser.parse_args()
    process_folder(args.input, args.output, args.sample_size)


if __name__ == "__main__":
    main()
